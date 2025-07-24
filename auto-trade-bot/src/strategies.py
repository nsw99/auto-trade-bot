import pandas as pd
import numpy as np
import ta
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import logging
from enum import Enum

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 데이터 클래스 및 열거형 정의 ---

class MarketRegime(Enum):
    """
    (개선 2) 시장 국면을 명확히 정의
    """
    STRONG_UPTREND = "강한 상승추세"
    WEAK_UPTREND = "약한 상승추세"
    SIDEWAYS_NEUTRAL = "중립 횡보"
    SIDEWAYS_VOLATILE = "변동성 횡보"
    STRONG_DOWNTREND = "강한 하락추세"
    NON_TRADEABLE = "거래 불가"

@dataclass
class TradeSignal:
    action: str
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)

@dataclass
class Holding:
    stock_code: str
    quantity: int
    avg_purchase_price: float
    # (개선 1) 동적 리스크 관리를 위한 필드 추가
    initial_stop_loss_price: float
    trailing_stop_loss_price: float
    first_profit_target: float
    is_first_profit_taken: bool = False

# --- 핵심 모듈: 포트폴리오, 시장분석, 전략 ---

class Portfolio:
    """자산, 현금, 포지션을 관리하는 포트폴리오 클래스"""
    def __init__(self, initial_cash: float = 10_000_000):
        self.initial_cash: float = initial_cash
        self.cash: float = initial_cash
        self.holdings: Dict[str, Holding] = {}
        self.total_value_history: List[Tuple[datetime, float]] = []
        self.trade_log: List[Dict] = []

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        holdings_value = sum(
            current_prices.get(code, h.avg_purchase_price) * h.quantity
            for code, h in self.holdings.items()
        )
        return self.cash + holdings_value

    def record_daily_value(self, timestamp: datetime, current_prices: Dict[str, float]):
        self.total_value_history.append((timestamp, self.get_total_value(current_prices)))

    def execute_trade(self, stock_code: str, action: str, price: float, quantity: int, timestamp: datetime, context: Dict):
        if action == 'buy':

            if self.cash < price * quantity: return
            self.cash -= price * quantity
            # (개선 4) 포지션 관리 시스템 강화
            self.holdings[stock_code] = Holding(
                stock_code=stock_code, quantity=quantity, avg_purchase_price=price,
                initial_stop_loss_price=context['stop_loss_price'],
                trailing_stop_loss_price=context['stop_loss_price'], # 초기 트레일링 스탑 = 초기 손절가
                first_profit_target=context['take_profit_price'],
            )
        elif action == 'sell':
            if stock_code not in self.holdings or self.holdings[stock_code].quantity < quantity: return
            self.cash += price * quantity
            if self.holdings[stock_code].quantity == quantity:
                del self.holdings[stock_code]
            else:
                self.holdings[stock_code].quantity -= quantity

        # (개선 5) 성과 추적을 위한 상세 정보 기록
        self.trade_log.append({**context, 'timestamp': timestamp, 'action': action, 'price': price, 'quantity': quantity})
        logger.info(f"TRADE: {timestamp.date()} - {action.upper()} {stock_code} ({quantity}주) @ {price:,.0f}원 | 이유: {context['reasons']}")

class MarketAnalyzer:
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> Dict:
        """모든 기술적 지표 계산"""
        indicators = {}
        close = data['close'].astype(float)
        # (개선 3) 다중 지표 조합을 위한 지표 추가
        indicators.update({
            'ma5': close.rolling(window=5).mean().iloc[-1], 'ma20': close.rolling(window=20).mean().iloc[-1],
            'ma60': close.rolling(window=60).mean().iloc[-1], 'rsi': ta.momentum.RSIIndicator(close).rsi().iloc[-1],
            'macd': ta.trend.MACD(close).macd_diff().iloc[-1], # MACD-Signal (히스토그램)
            'atr': ta.volatility.AverageTrueRange(data['high'], data['low'], close).average_true_range().iloc[-1],
            'bb_upper': ta.volatility.BollingerBands(close).bollinger_hband().iloc[-1],
            'bb_lower': ta.volatility.BollingerBands(close).bollinger_lband().iloc[-1],
            'volume_ratio': data['volume'].iloc[-1] / data['volume'].rolling(window=20).mean().iloc[-1]
        })
        return indicators

    @staticmethod
    def detect_market_regime(data: pd.DataFrame) -> MarketRegime:
        """
        (개선 2) 시장 적응성 강화를 위한 국면 자동 인식
        """
        if len(data) < 60: return MarketRegime.NON_TRADEABLE

        close = data['close'].astype(float)
        adx = ta.trend.ADXIndicator(data['high'], data['low'], close).adx().iloc[-1]
        volatility = close.pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) # 연환산 변동성

        is_uptrend = data['close'].iloc[-1] > data['close'].rolling(window=60).mean().iloc[-1]

        if adx > 25 and is_uptrend: return MarketRegime.STRONG_UPTREND
        if adx > 20 and is_uptrend: return MarketRegime.WEAK_UPTREND
        if adx > 25 and not is_uptrend: return MarketRegime.STRONG_DOWNTREND
        if volatility > 0.4: return MarketRegime.SIDEWAYS_VOLATILE # 연 40% 이상 변동성
        return MarketRegime.SIDEWAYS_NEUTRAL

class StrategyEvaluator:
    @staticmethod
    def calculate_position_size(total_capital: float, risk_per_trade: float, stop_loss_distance: float) -> int:
        """
        (개선 4) 변동성 고려 투자 금액 조정
        """
        if stop_loss_distance <= 0: return 0
        dollar_risk = total_capital * risk_per_trade
        return int(dollar_risk // stop_loss_distance)

    @staticmethod
    def evaluate_buy_signal(indicators: Dict, current_price: float) -> TradeSignal:
        """
        (개선 3) 진입 신호 정확도 향상 (컨플루언스)
        """
        conditions = {
            "RSI 과매도 근접": indicators['rsi'] < 40,
            "단기 추세 상승": indicators['ma5'] > indicators['ma20'],
            "장기 추세 상승": current_price > indicators['ma60'],
            "MACD 상승 전환": indicators['macd'] > 0,
            "볼린저 하단 근접": current_price < indicators['bb_lower'] * 1.05, # 하단 5% 이내
        }

        true_conditions = [reason for reason, is_true in conditions.items() if is_true]

        # 최소 3개 이상의 조건이 충족되고, 거래량 필터를 통과해야 함
        if len(true_conditions) >= 3 and indicators['volume_ratio'] > 1.2:
            confidence = 0.5 + (len(true_conditions) - 3) * 0.15 # 3개 0.5, 4개 0.65, 5개 0.8
            reasons = true_conditions + [f"거래량 증가({indicators['volume_ratio']:.1f}x)"]
            return TradeSignal(action='buy', confidence=confidence, reasons=reasons)

        return TradeSignal(action='hold')

    @staticmethod
    def evaluate_sell_signal(holding: Holding, current_price: float) -> Optional[Tuple[str, float, str]]:
        """
        (개선 1) 다단계 익절/손절 로직
        """
        # 1. 트레일링 스탑 체크 (수익 확보 후)
        if holding.is_first_profit_taken:
            if current_price < holding.trailing_stop_loss_price:
                return 'sell', holding.quantity, f"트레일링 스탑"

        # 2. 초기 손절 라인 체크
        if current_price < holding.initial_stop_loss_price:
            return 'sell', holding.quantity, f"초기 손절"

        # 3. 1차 익절 체크
        if not holding.is_first_profit_taken and current_price >= holding.first_profit_target:
            return 'sell', holding.quantity // 2, f"1차 익절" # 50% 분할 매도

        return None

# --- 백테스팅 엔진 ---

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_cash: float = 10_000_000):
        self.data = data
        self.portfolio = Portfolio(initial_cash)
        self.stock_code = "TEST_STOCK"
        self.regime_multiplier = { # 국면별 포지션 사이징
            MarketRegime.STRONG_UPTREND: 1.2, MarketRegime.WEAK_UPTREND: 1.0,
            MarketRegime.SIDEWAYS_NEUTRAL: 0.5, MarketRegime.SIDEWAYS_VOLATILE: 0.3,
            MarketRegime.STRONG_DOWNTREND: 0.0, MarketRegime.NON_TRADEABLE: 0.0
        }

    def run(self):
        logger.info("백테스팅을 시작합니다...")
        for i in range(60, len(self.data)):
            current_data_window = self.data.iloc[:i+1]
            current_bar = self.data.iloc[i]
            current_price = current_bar['close']
            current_timestamp = current_bar.name

            self.portfolio.record_daily_value(current_timestamp, {self.stock_code: current_price})

            indicators = MarketAnalyzer.calculate_indicators(current_data_window)
            market_regime = MarketAnalyzer.detect_market_regime(current_data_window)

            # --- 매도 로직 ---
            if self.stock_code in self.portfolio.holdings:
                holding = self.portfolio.holdings[self.stock_code]
                sell_decision = StrategyEvaluator.evaluate_sell_signal(holding, current_price)
                if sell_decision:
                    action, quantity_to_sell, reason = sell_decision
                    context = {'reasons': [reason], 'market_regime': market_regime.value}
                    self.portfolio.execute_trade(self.stock_code, action, current_price, quantity_to_sell, current_timestamp, context)
                    # 1차 익절 후 상태 업데이트
                    if reason == "1차 익절":
                        holding.is_first_profit_taken = True
                        holding.trailing_stop_loss_price = holding.avg_purchase_price # 손절가를 본전으로

                # 트레일링 스탑 가격 업데이트
                if holding.is_first_profit_taken:
                    new_trailing_stop = current_price - indicators['atr'] * 2.5
                    holding.trailing_stop_loss_price = max(holding.trailing_stop_loss_price, new_trailing_stop)

            # --- 매수 로직 ---
            regime_mult = self.regime_multiplier.get(market_regime, 0.0)
            if regime_mult > 0 and self.stock_code not in self.portfolio.holdings:
                buy_signal = StrategyEvaluator.evaluate_buy_signal(indicators, current_price)
                if buy_signal.action == 'buy':
                    stop_loss_price = current_price - (indicators['atr'] * 2)
                    stop_loss_distance = current_price - stop_loss_price

                    risk_per_trade = 0.02 * regime_mult * buy_signal.confidence # 국면과 신뢰도에 따라 리스크 조절

                    quantity_to_buy = StrategyEvaluator.calculate_position_size(
                        self.portfolio.get_total_value({self.stock_code: current_price}),
                        risk_per_trade,
                        stop_loss_distance
                    )

                    if quantity_to_buy > 0:
                        context = {
                            'reasons': buy_signal.reasons, 'confidence': buy_signal.confidence,
                            'market_regime': market_regime.value,
                            'stop_loss_price': stop_loss_price,
                            'take_profit_price': current_price + (stop_loss_distance * 2) # 손익비 1:2
                        }
                        self.portfolio.execute_trade(self.stock_code, 'buy', current_price, quantity_to_buy, current_timestamp, context)

        logger.info("백테스팅이 종료되었습니다.")
        return self.generate_performance_report()

    def generate_performance_report(self) -> Dict:
        # (이전 코드와 동일, 생략)
        equity_curve = pd.Series([val for _, val in self.portfolio.total_value_history], index=[d for d, _ in self.portfolio.total_value_history])
        initial_value = self.portfolio.initial_cash
        final_value = equity_curve.iloc[-1]

        total_return = (final_value / initial_value - 1) * 100

        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100

        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

        # 소르티노 비율 계산 추가
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()
        sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_std if downside_std > 0 else float('inf')

        report = {
            "기간": f"{self.data.index[0].date()} ~ {self.data.index[-1].date()}",
            "최종 자산 / 총 수익률 (%)": f"{final_value:,.0f} 원 / {total_return:.2f}%",
            "최대 낙폭 (MDD) (%)": f"{max_drawdown:.2f}",
            "샤프 비율 (연환산)": f"{sharpe_ratio:.2f}",
            "소르티노 비율 (연환산)": f"{sortino_ratio:.2f}",
            "총 거래 횟수": len(self.portfolio.trade_log)
        }
        logger.info("\n--- 최종 성과 보고서 ---\n" + "\n".join([f"{key}: {value}" for key, value in report.items()]))
        return report

# --- 사용 예시 ---
if __name__ == '__main__':
    # 가상 데이터 대신 실제 데이터 로드를 권장 (yfinance 라이브러리 예시)
    try:
        import yfinance as yf
        logger.info("삼성전자(005930.KS) 데이터를 다운로드합니다...")
        ohlc_data = yf.download('005930.KS', start='2020-01-01', end='2024-12-31')
        ohlc_data.columns = [col.lower() for col in ohlc_data.columns] # 컬럼명 소문자로 변경

        if ohlc_data.empty: raise Exception("데이터 다운로드 실패")

        backtester = Backtester(data=ohlc_data, initial_cash=10_000_000)
        final_report = backtester.run()

    except Exception as e:
        logger.error(f"실행 오류: {e}. 가상 데이터로 테스트합니다.")
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
        price_data = 10000 + np.random.randn(len(dates)).cumsum() * 100
        ohlc_data = pd.DataFrame({
            'open': price_data, 'high': price_data + 100, 'low': price_data - 100,
            'close': price_data + np.random.uniform(-50, 50, size=len(dates)),
            'volume': np.random.randint(100000, 5000000, size=len(dates))
        }, index=dates)
        backtester = Backtester(data=ohlc_data, initial_cash=10_000_000)
        final_report = backtester.run()