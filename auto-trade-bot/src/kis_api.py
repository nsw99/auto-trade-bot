import os
from datetime import datetime, timedelta
import time
import pandas as pd
from dotenv import load_dotenv
from pykis import PyKis, KisAuth, KisBalance, KisAccount
from decimal import Decimal, InvalidOperation

from src.logger import logger
from dataclasses import dataclass, field

@dataclass
class DepositData:
    """Mutable deposit data."""
    amount: Decimal
    exchange_rate: Decimal
    currency: str

@dataclass
class StockData:
    """Mutable stock balance data."""
    symbol: str
    market: str
    qty: int
    price: Decimal
    amount: Decimal
    profit: Decimal
    profit_rate: Decimal

@dataclass
class BalanceData:
    """A mutable container for the processed balance information."""
    purchase_amount: Decimal
    current_amount: Decimal
    profit: Decimal
    profit_rate: Decimal
    deposits: dict[str, DepositData] = field(default_factory=dict)
    stocks: list[StockData] = field(default_factory=list)



class KISApiHandler:
    """
    python-kis 라이브러리를 감싸고, API 호출을 관리하는 핸들러 클래스.
    """
    def __init__(self, config_loader):
        self.config = config_loader.get_config()
        self.kis: PyKis = self._initialize_kis()
        self.stock_info_cache = {}
        self.trading_hours_cache = {}
        self.exchange_rate_cache = {"timestamp": 0, "rate": None}

    def _to_decimal(self, value, default=Decimal('0')):
        """숫자나 문자열을 Decimal로 안전하게 변환합니다."""
        if value is None:
            return default
        try:
            # str()로 감싸서 float의 부정확한 표현을 피함
            return Decimal(str(value))
        except (InvalidOperation, TypeError):
            return default

    

    def _initialize_kis(self) -> PyKis:
        """KIS API 클라이언트를 초기화합니다."""
        load_dotenv()

        # 실계좌 정보 로드
        app_key = os.getenv('KIS_APP_KEY')
        app_secret = os.getenv('KIS_APP_SECRET')
        account_no = os.getenv('KIS_ACCOUNT_NO')
        hts_id = os.getenv("HIS_ID")

        # 모의투자 계좌 정보 로드
        virtual_app_key = os.getenv('VIRTUAL_KIS_APP_KEY')
        virtual_app_secret = os.getenv('VIRTUAL_KIS_APP_SECRET')
        virtual_account_no = os.getenv('VIRTUAL_KIS_ACCOUNT_NO')

        if not all([app_key, app_secret, account_no, hts_id, virtual_app_key, virtual_app_secret, virtual_account_no]):
            raise ValueError(".env 파일에서 모든 API 자격증명(실계좌/모의투자)을 찾을 수 없습니다.")

        try:
            # 실계좌 인증 정보 저장
            real_auth = KisAuth(
                id=hts_id,
                appkey=app_key,
                secretkey=app_secret,
                account=account_no,
                virtual=False,
            )
            real_auth.save("secret.json")
            
            # 모의투자 인증 정보 저장
            virtual_auth = KisAuth(
                id=hts_id,
                appkey=virtual_app_key,
                secretkey=virtual_app_secret,
                account=virtual_account_no,
                virtual=True,
            )
            virtual_auth.save("virtual_secret.json")
            
            # 실계좌와 모의투자 정보를 모두 사용하여 PyKis 클라이언트 초기화
            kis = PyKis(
                KisAuth.load("secret.json"),
                KisAuth.load("virtual_secret.json"),
                keep_token=True
            )
            
            # 설정 파일(config.json)에 따라 사용할 계좌를 활성화
            # account_type = self.config.get('account_type', 'virtual')
            # kis.change_account(account_type)

            logger.info(f"{'모의투자' if 'virtual' == 'virtual' else '실전투자'} 계좌에 대해 KIS API를 초기화했습니다.")
            return kis
        except Exception as e:
            logger.error(f"KIS API 초기화 실패: {e}")
            raise

    def get_stock_name(self, stock_code: str) -> str:
        """종목 코드로 종목명을 조회하고 캐시에 저장합니다."""
        if stock_code in self.stock_info_cache:
            return self.stock_info_cache[stock_code]
        
        time.sleep(0.2) # API Rate Limit
        try:
            stock_name = self.kis.stock(stock_code).name
            if stock_name:
                self.stock_info_cache[stock_code] = stock_name
                logger.info(f"종목명 조회 성공: {stock_code} -> {stock_name}")
                return stock_name
            return stock_code
        except Exception as e:
            logger.error(f"{stock_code} 종목명 조회 실패: {e}")
            return stock_code

    def fetch_balance(self) -> BalanceData | None:
        """현재 계좌 잔고 객체를 조회하여 금액 관련 필드를 Decimal로 변환 후 반환합니다."""
        time.sleep(0.2)
        try:
            account = self.kis.account()
            balance = account.balance()

            # Create a new mutable BalanceData object
            processed_balance = BalanceData(
                purchase_amount=self._to_decimal(balance.purchase_amount),
                current_amount=self._to_decimal(balance.current_amount),
                profit=self._to_decimal(balance.profit),
                profit_rate=self._to_decimal(balance.profit_rate)
            )

            # Process deposits
            for currency, deposit in balance.deposits.items():
                processed_balance.deposits[currency] = DepositData(
                    amount=self._to_decimal(deposit.amount),
                    exchange_rate=self._to_decimal(deposit.exchange_rate),
                    currency=currency
                )

            # Process stocks
            if hasattr(balance, 'stocks') and balance.stocks:
                for stock in balance.stocks:
                    processed_stock = StockData(
                        symbol=stock.symbol,
                        market=getattr(stock, 'market', ''),
                        qty=stock.qty,
                        price=self._to_decimal(stock.price),
                        amount=self._to_decimal(stock.amount),
                        profit=self._to_decimal(getattr(stock, 'profit', 0)),
                        profit_rate=self._to_decimal(getattr(stock, 'profit_rate', 0))
                    )
                    processed_balance.stocks.append(processed_stock)
            
            logger.info("계좌 잔고 조회 및 처리 완료.")
            return processed_balance
            
        except Exception as e:
            logger.error(f"계좌 잔고 조회 실패: {e}", exc_info=True)
            return None

    def fetch_current_price(self, stock_code: str) -> Decimal:
        """종목의 현재가를 조회하여 Decimal 타입으로 반환합니다."""
        time.sleep(0.2)
        try:
            price = self.kis.stock(stock_code).quote().price
            decimal_price = self._to_decimal(price)
            logger.info(f"'{self.get_stock_name(stock_code)}({stock_code})' 현재가: {decimal_price:,.0f}원")
            return decimal_price
        except Exception as e:
            logger.error(f"{stock_code} 현재가 조회 실패: {e}")
            return Decimal('0')

    def search_stock(self, keyword: str) -> list[dict]:
        """키워드로 국내외 주식을 검색하여 상세 정보를 반환합니다."""
        time.sleep(0.2)
        try:
            # pykis는 종목 코드로만 검색을 지원하므로, fetch_price_v2를 사용.
            # 실제 키워드 검색 기능이 필요하다면 별도 API가 필요하지만,
            # 현재 구조에서는 종목 코드를 키워드로 간주.
            stock_info = self.fetch_price_v2(keyword)
            if stock_info:
                return [stock_info]
            return []
        except Exception as e:
            logger.error(f"'{keyword}' 주식 검색 실패: {e}", exc_info=True)
            return []

    def fetch_price_v2(self, stock_code: str) -> dict | None:
        """종목의 상세 정보(심볼, 이름, 현재가, 마켓)를 조회합니다. (대시보드용)"""
        time.sleep(0.2)
        try:
            quote = self.kis.stock(stock_code).quote()
            if not quote:
                return None

            market = getattr(quote, 'market', '').upper()
            price = self._to_decimal(getattr(quote, 'price', 0))
            
            result = {
                'symbol': getattr(quote, 'symbol', stock_code),
                'name': getattr(quote, 'name', 'N/A'),
                'market': market,
                'price': price, # Decimal 타입 유지
                'currency': 'KRW'
            }

            if market not in ['KRX', 'KOSPI', 'KOSDAQ'] and market != '':
                usd_exchange_rate = self.fetch_exchange_rate('USD')
                result['currency'] = 'USD'
                result['price_krw'] = price * usd_exchange_rate
            
            return result

        except Exception as e:
            logger.error(f"'{stock_code}' 상세 정보 조회 실패: {e}")
            return None

    def fetch_ohlcv(self, stock_code: str, days: int = 365) -> pd.DataFrame:
        """
        지정된 종목의 과거 OHLCV 데이터를 DataFrame으로 조회합니다.
        가격 데이터는 Decimal로 변환됩니다.
        """
        try:
            stock = self.kis.stock(stock_code)
            start_date = datetime.now().date() - timedelta(days=days)
            chart = stock.chart(period='day', start=start_date)
            df = chart.df()

            if df.empty:
                logger.warning(f"'{stock_code}'에 대한 차트 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()

            # 가격 관련 컬럼을 float으로 변환
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
            df.sort_values(by='time', inplace=True)

            return df

        except Exception as e:
            logger.error(f"'{stock_code}' OHLCV 데이터 조회 중 오류 발생: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_exchange_rate(self, currency: str = 'USD') -> Decimal:
        """
        지정된 통화의 현재 환율을 조회하고 캐시합니다. (10분 캐시)
        """
        now = time.time()
        cache = self.exchange_rate_cache.get(currency)

        if cache and (now - cache["timestamp"] < 600): # 10분
            return cache["rate"]

        logger.info(f"'{currency}' 환율 정보를 API를 통해 조회합니다.")
        try:
            balance = self.fetch_balance()
            if balance and currency in balance.deposits:
                exchange_rate = self._to_decimal(balance.deposits[currency].exchange_rate)
                if exchange_rate > 0:
                    self.exchange_rate_cache[currency] = {"rate": exchange_rate, "timestamp": now}
                    logger.info(f"'{currency}' 환율 정보 조회 성공: {exchange_rate}")
                    return exchange_rate
        except Exception as e:
            logger.error(f"환율 조회 중 오류 발생: {e}", exc_info=True)

        if cache: # API 실패 시 만료된 캐시라도 반환
            logger.warning(f"'{currency}' 환율 정보 갱신 실패. 캐시된 값 사용: {cache['rate']}")
            return cache["rate"]
            
        logger.error(f"'{currency}' 환율 정보를 조회할 수 없습니다. 기본값 1.0을 반환합니다.")
        return Decimal('1.0')

    def fetch_trading_hours(self, market: str):
        """지정된 시장의 거래 시간을 조회하고 캐시합니다."""
        if market in self.trading_hours_cache:
            return self.trading_hours_cache[market]

        try:
            time.sleep(0.2) # API Rate Limit
            logger.info(f"'{market}' 시장의 거래 시간 정보를 API를 통해 조회합니다.")
            trading_hours = self.kis.trading_hours(market)
            self.trading_hours_cache[market] = trading_hours
            return trading_hours
        except Exception as e:
            logger.error(f"'{market}' 시장 거래 시간 조회 실패: {e}")
            return None

    def buy(self, stock_code: str, amount: Decimal):
        """주어진 금액(Decimal)만큼 시장가 매수 주문을 실행합니다."""
        stock_name = self.get_stock_name(stock_code)
        try:
            current_price = self.fetch_current_price(stock_code)
            if current_price <= Decimal('0'):
                logger.error(f"가격 정보를 확인할 수 없어 '{stock_name}({stock_code})' 주문을 진행할 수 없습니다.")
                return None

            # 모든 계산을 Decimal로 수행
            quantity = int(amount // current_price)
            if quantity == 0:
                logger.warning(f"주문 금액 {amount:,.0f}원이 '{stock_name}' 1주 가격({current_price:,.0f}원)보다 작아 주문할 수 없습니다.")
                return None

            logger.info(f"'{stock_name}({stock_code})'에 대한 시장가 매수 주문을 전송합니다: {quantity}주.")
            order_result = self.kis.stock(stock_code).buy(qty=quantity)
            
            order_result.quantity = quantity
            order_result.price = current_price # Decimal 타입 유지
            return order_result
        except Exception as e:
            logger.error(f"'{stock_name}({stock_code})' 시장가 매수 주문 실패: {e}", exc_info=True)
            return None

    def sell(self, stock_code: str, quantity: int):
        """주어진 수량만큼 시장가 매도 주문을 실행합니다."""
        stock_name = self.get_stock_name(stock_code)
        logger.info(f"'{stock_name}({stock_code})'에 대한 시장가 매도 주문을 전송합니다: {quantity}주.")
        try:
            order_result = self.kis.stock(stock_code).sell(qty=quantity)
            
            order_result.price = self.fetch_current_price(stock_code) # Decimal 타입
            order_result.quantity = quantity
            return order_result
        except Exception as e:
            logger.error(f"'{stock_name}({stock_code})' 시장가 매도 실패: {e}", exc_info=True)
            return None
