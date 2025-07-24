import time
import schedule
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pykis import KisSubscriptionEventArgs, KisRealtimePrice

from src.logger import logger
from src.config_loader import ConfigLoader
from src.kis_api import KISApiHandler
from src.strategies import StrategyEvaluator, MarketAnalyzer, Holding
from src.purchase_logger import log_purchase
from src.redis_handler import redis_handler

class DCABot:
    def __init__(self, config_path='config.json'):
        logger.info("DCA 봇을 초기화합니다...")
        if not redis_handler:
            raise ConnectionError("Redis 핸들러가 초기화되지 않았습니다. 봇을 종료합니다.")
            
        self.config_loader = ConfigLoader(config_path)
        self.api_handler = KISApiHandler(self.config_loader)
        self.strategy_evaluator = StrategyEvaluator()
        self.active_plans = self.config_loader.get_active_plans()
        logger.info(f"활성화된 DCA 플랜: {len(self.active_plans)}개")
        
        self.jobs = []
        self.monthly_spending = {}
        self.portfolio = {}
        self.account_balance = None
        
        self._initialize_portfolio()
        self._setup_scheduler()
        self._subscribe_to_realtime_data()

    def _to_decimal(self, value, default=Decimal('0')):
        """Helper to safely convert to Decimal."""
        if value is None:
            return default
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError):
            return default

    def _on_realtime_price(self, sender, e: KisSubscriptionEventArgs[KisRealtimePrice]):
        """실시간 체결가 수신 콜백 함수."""
        redis_handler.r.hset('realtime_prices', e.response.symbol, str(e.response.price))

    def _subscribe_to_realtime_data(self):
        """활성 플랜의 모든 종목에 대해 실시간 데이터 구독을 시작합니다."""
        if not self.active_plans:
            logger.info("활성화된 플랜이 없어 실시간 데이터 구독을 시작하지 않습니다.")
            return
            
        stock_codes_to_subscribe = list(set(plan['stock_code'] for plan in self.active_plans))
        
        if not stock_codes_to_subscribe:
            logger.info("구독할 종목이 없어 실시간 데이터 구독을 시작하지 않습니다.")
            return

        logger.info(f"{len(stock_codes_to_subscribe)}개 종목에 대한 실시간 가격 구독을 시작합니다: {stock_codes_to_subscribe}")
        for code in stock_codes_to_subscribe:
            self.api_handler.kis.stock(code).on('price', self._on_realtime_price)

    def _initialize_portfolio(self):
        """잔고를 기반으로 포트폴리오와 계좌 요약을 초기화하고 KPI를 업데이트합니다."""
        self.account_balance = self.api_handler.fetch_balance()
        if self.account_balance and self.account_balance.stocks:
            self.portfolio = {stock.symbol: vars(stock) for stock in self.account_balance.stocks}
        else:
            self.portfolio = {}
        
        cash = self._get_cash_balance()
        logger.info(f"포트폴리오 초기화 완료. {len(self.portfolio)}개 종목 보유, 예수금: {cash:,.0f}원")
        self.update_portfolio_kpis()

    def _get_cash_balance(self) -> Decimal:
        """사용 가능한 현금 잔고(예수금)를 Decimal로 반환합니다."""
        if self.account_balance is None:
            logger.warning("계좌 조회 실패.")
            return Decimal('0')

        if hasattr(self.account_balance, 'deposits') and isinstance(self.account_balance.deposits, dict) and 'KRW' in self.account_balance.deposits:
            krw_deposit = self.account_balance.deposits['KRW']
            if hasattr(krw_deposit, 'amount'):
                return self._to_decimal(krw_deposit.amount)
        
        logger.warning("계좌 잔고에서 KRW 예치금 정보를 찾을 수 없습니다.")
        return Decimal('0')

    def update_portfolio_kpis(self):
        """포트폴리오 정보를 기반으로 KPI를 계산하고 Redis에 업데이트합니다."""
        if not self.account_balance:
            return

        kpi_data = {
            "total_purchase_amount": self.account_balance.purchase_amount,
            "total_eval_amount": self.account_balance.current_amount,
            "pnl": self.account_balance.profit,
            "pnl_rate": self.account_balance.profit_rate,
            "cash_balance": self._get_cash_balance(),
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        redis_handler.update_kpis(kpi_data)
        
        log_kpi = {k: (f"{v:,.2f}" if isinstance(v, Decimal) else v) for k, v in kpi_data.items()}
        logger.info(f"포트폴리오 KPI가 업데이트되었습니다: {log_kpi}")

    def _get_current_year_month(self):
        return datetime.now().strftime('%Y-%m')

    def _get_remaining_budget(self, plan) -> Decimal:
        if 'monthly_budget' not in plan:
            return Decimal('inf')

        plan_name = plan['plan_name']
        total_budget = self._to_decimal(plan['monthly_budget'])
        current_month = self._get_current_year_month()

        if plan_name not in self.monthly_spending or self.monthly_spending[plan_name]['month'] != current_month:
            self.monthly_spending[plan_name] = {'month': current_month, 'spent': Decimal('0')}
            logger.info(f"'{plan_name}' 계획에 대한 월별 지출을 초기화합니다. ({current_month})")

        return total_budget - self.monthly_spending[plan_name]['spent']

    def _update_spending(self, plan, amount_spent: Decimal):
        if 'monthly_budget' in plan:
            plan_name = plan['plan_name']
            self.monthly_spending[plan_name]['spent'] += amount_spent
            logger.info(f"'{plan_name}' 계획 지출 업데이트: {amount_spent:,.0f}원. (이번 달 총 지출: {self.monthly_spending[plan_name]['spent']:,.0f}원)")

    def _setup_scheduler(self):
        logger.info(f"{len(self.active_plans)}개의 활성화된 계획에 대한 스케줄을 설정합니다.")
        for plan in self.active_plans:
            stock_name = self.api_handler.get_stock_name(plan['stock_code'])
            plan['display_name'] = f"{stock_name} ({plan['plan_name']})"
            mode = plan.get('mode')

            if mode in ['auto_timing', 'full_auto']:
                job = schedule.every(1).minutes.do(self.process_plan, plan)
                self.jobs.append(job)
                logger.info(f"'{plan['display_name']}' 계획에 대한 매매 평가를 1분 간격으로 예약했습니다.")
            else:
                logger.warning(f"'{plan['display_name']}' 계획은 알 수 없는 모드({mode})를 가지고 있습니다.")

    def _is_market_open(self, market: str):
        trading_hours = self.api_handler.fetch_trading_hours(market)
        if not trading_hours:
            logger.warning(f"'{market}' 시장의 거래 시간을 확인할 수 없어, 열린 것으로 간주하고 진행합니다.")
            return True

        try:
            now_time = datetime.now().time()
            open_time = trading_hours.open_kst
            close_time = trading_hours.close_kst
            if open_time < close_time:
                return open_time <= now_time <= close_time
            else:
                return now_time >= open_time or now_time <= close_time
        except Exception as e:
            logger.error(f"'{market}' 시장 시간 파싱 오류: {e}")
            return False

    def process_plan(self, plan):
        plan_display_name = plan.get('display_name', plan['plan_name'])
        stock_code = plan['stock_code']
        market = plan.get('market', 'KR')

        logger.info(f"--- '{plan_display_name}'({stock_code}/{market}) 플랜 처리 시작 ---")

        if not self._is_market_open(market):
            logger.info(f"'{market}' 시장이 열리는 시간이 아니므로 매매 평가를 건너뜁니다.")
            return

        self._initialize_portfolio()

        if stock_code in self.portfolio:
            self.check_and_execute_sell(plan, self.portfolio[stock_code])

        self.check_and_execute_buy(plan)

    def check_and_execute_buy(self, plan):
        plan_display_name = plan.get('display_name', plan['plan_name'])
        
        remaining_monthly_budget = self._get_remaining_budget(plan)
        if 'monthly_budget' in plan and remaining_monthly_budget <= 0:
            logger.info(f"'{plan_display_name}' 플랜은 이번 달 예산을 모두 소진하여 매수를 건너뜁니다.")
            return

        ohlcv_df = self.api_handler.fetch_ohlcv(plan['stock_code'])
        if ohlcv_df.empty:
            logger.warning(f"OHLCV 데이터 조회 실패로 '{plan_display_name}' 매수 평가를 건너뜁니다.")
            return

        indicators = MarketAnalyzer.calculate_indicators(ohlcv_df)
        if not indicators:
            logger.warning(f"지표 계산 실패로 '{plan_display_name}' 매수 평가를 건너뜁니다.")
            return
        
        current_price = self.api_handler.fetch_current_price(plan['stock_code'])
        if current_price <= Decimal('0'):
            logger.warning(f"'{plan_display_name}' 현재가를 조회할 수 없어 매수 평가를 건너뜁니다.")
            return

        buy_signal = self.strategy_evaluator.evaluate_buy_signal(indicators, float(current_price))
        should_buy, reason = (buy_signal.action == 'buy'), ", ".join(buy_signal.reasons)

        if should_buy:
            logger.info(f"매수 신호 발생 '{plan_display_name}'. 이유: {reason}")
            
            base_tranche = self._to_decimal(plan.get('buy_amount_per_trade', 50000))
            cash_balance = self._get_cash_balance()
            
            amount_to_buy = min(base_tranche, remaining_monthly_budget, cash_balance)

            if amount_to_buy > Decimal('1000'):
                order_result = self.api_handler.buy(plan['stock_code'], amount_to_buy)
                if order_result:
                    self._log_transaction('buy', plan, order_result, reason)
            else:
                logger.error(f"'{plan_display_name}' 유효 주문 금액 부족. (주문 시도액: {amount_to_buy:,.0f}원, 현금 잔고: {cash_balance:,.0f}원)")
        else:
            logger.info(f"매수 보류 신호 '{plan_display_name}'. 이유: {reason}")

    def check_and_execute_sell(self, plan, stock_details):
        plan_display_name = plan.get('display_name', plan['plan_name'])
        stock_code = plan['stock_code']
        sell_strategy_config = plan.get('sell_strategy', {'enabled': False})

        if not sell_strategy_config.get('enabled'):
            return

        logger.info(f"'{plan_display_name}' 보유 주식에 대한 매도 평가를 시작합니다.")
        
        current_price = self.api_handler.fetch_current_price(stock_code)
        if current_price <= Decimal('0'):
            logger.warning(f"'{plan_display_name}' 현재가를 조회할 수 없어 매도 평가를 건너뜁니다.")
            return

        ohlcv_df = self.api_handler.fetch_ohlcv(stock_code)
        if ohlcv_df.empty:
            logger.warning(f"OHLCV 데이터 조회 실패로 '{plan_display_name}' 매도 평가를 건너뜁니다.")
            return

        indicators = MarketAnalyzer.calculate_indicators(ohlcv_df)
        if not indicators:
            logger.warning(f"지표 계산 실패로 '{plan_display_name}' 매도 평가를 건너뜁니다.")
            return

        avg_purchase_price = float(self._to_decimal(stock_details.get('price', '0')))
        quantity = int(stock_details.get('qty', 0))

        if quantity == 0:
            return
        
        # Create a Holding object for evaluation
        holding_info = Holding(
            stock_code=stock_code,
            quantity=quantity,
            avg_purchase_price=avg_purchase_price,
            initial_stop_loss_price=avg_purchase_price * 0.9, # Example: 10% stop-loss
            trailing_stop_loss_price=avg_purchase_price * 0.9,
            first_profit_target=avg_purchase_price * 1.2, # Example: 20% profit target
            is_first_profit_taken=False
        )

        sell_decision = self.strategy_evaluator.evaluate_sell_signal(holding_info, float(current_price))
        
        should_sell, reason = (sell_decision is not None), ""
        if should_sell:
            _, _, reason = sell_decision

        if should_sell:
            logger.info(f"매도 신호 발생 '{plan_display_name}'. 이유: {reason}")
            order_result = self.api_handler.sell(stock_code, quantity)
            if order_result:
                self._log_transaction('sell', plan, order_result, reason)
        else:
            logger.info(f"매도 보류 신호 '{plan_display_name}'. 이유: {reason}")

    def _log_transaction(self, tx_type, plan, order_result, reason):
        stock_code = plan['stock_code']
        stock_name = self.api_handler.get_stock_name(stock_code)
        quantity = order_result.quantity
        price = order_result.price # Decimal
        total_amount = price * Decimal(quantity)
        
        log_details = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': tx_type,
            'plan_name': plan['plan_name'],
            'stock_code': stock_code,
            'stock_name': stock_name,
            'quantity': quantity, 
            'price': price, 
            'amount': total_amount,
            'reason': reason,
        }
        
        log_purchase(log_details, tx_type)
        redis_handler.store_trade(log_details)
        
        if tx_type == 'buy':
            self._update_spending(plan, total_amount)

        logger.info(f"[{tx_type.upper()}] {stock_name} {quantity}주, {price:,.0f}원. 사유: {reason}")

    def run(self):
        """봇의 메인 루프를 시작합니다."""
        logger.info("DCA 봇이 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.")
        self.update_portfolio_kpis()
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("봇을 종료합니다.")
