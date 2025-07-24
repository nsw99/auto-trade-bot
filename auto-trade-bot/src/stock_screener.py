import pandas as pd
from src.kis_api import KISApiHandler
from src.config_loader import ConfigLoader
import os
from decimal import Decimal, InvalidOperation
from pykrx import stock
from datetime import datetime, timedelta
import concurrent.futures
from src.logger import logger
import yfinance as yf

def get_kosdaq100_codes() -> list[str]:
    """
    pykrx를 사용하여 코스닥 시가총액 상위 100개 종목 코드 리스트를 조회합니다.
    휴일에도 데이터를 가져올 수 있도록 예외 처리를 추가했습니다.
    """
    for i in range(5):
        try:
            target_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            df = stock.get_market_cap_by_ticker(date=target_date, market="KOSDAQ")
            df = df.sort_values(by='시가총액', ascending=False)
            logger.info(f"{target_date} 기준 코스닥 10 리스트 조회 성공")
            return df.head(10).index.tolist()
        except Exception:
            logger.warning(f"{target_date} 데이터 조회 실패. 하루 전 날짜로 재시도합니다.")
            continue
    logger.error("최근 5일간 코스닥 데이터를 가져오지 못했습니다.")
    return []

def get_nasdaq100_codes() -> list[str]:
    """
    Wikipedia에서 나스닥 100 종목 코드 리스트를 조회합니다.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        nasdaq100_symbols = []
        for table in tables:
            if 'Ticker' in table.columns:
                nasdaq100_symbols = table['Ticker'].tolist()
                break
        return nasdaq100_symbols
    except Exception as e:
        print(f"나스닥 100 종목을 가져오는 중 오류 발생: {e}")
        return []

class StockScreener:
    def __init__(self, api_handler: KISApiHandler, config_loader: ConfigLoader):
        self.api = api_handler
        self.qvm_rules = config_loader.get_qvm_rules()
        logger.info("StockScreener 초기화 완료. QVM 규칙이 로드되었습니다.")

    def _get_financials_pykrx(self, code: str, date: str) -> dict | None:
        """pykrx를 사용하여 국내 주식의 재무 정보를 가져옵니다."""
        try:
            df = stock.get_market_fundamental(date, date, code)
            if df.empty:
                return None

            fund = df.iloc[0]
            price = stock.get_market_ohlcv(date, date, code).iloc[0]['종가']
            
            # ROE 계산 (BPS와 EPS 기반)
            bps = fund.get('BPS', 0)
            eps = fund.get('EPS', 0)
            roe = (eps / bps) * 100 if bps > 0 else 0

            # PSR 조회를 위해 yfinance 보조 사용
            ticker = yf.Ticker(f"{code}.KS")
            psr = ticker.info.get('priceToSalesTrailing12Months')

            return {
                'code': code,
                'name': stock.get_market_ticker_name(code),
                'price': price,
                'per': fund.get('PER'),
                'pbr': fund.get('PBR'),
                'psr': psr,
                'roe': roe,
                'roe_history': [] # pykrx로는 과거 데이터 조회가 어려워 비워둠
            }
        except Exception as e:
            logger.warning(f"{code} (pykrx) 정보 조회 실패: {e}", exc_info=True)
            return None

    def _get_financials_yfinance(self, code: str) -> dict | None:
        """yfinance를 사용하여 해외 주식의 재무 정보를 가져옵니다."""
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            
            price = info.get('currentPrice')
            per = info.get('trailingPE')
            pbr = info.get('priceToBook')
            psr = info.get('priceToSalesTrailing12Months')
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0

            if any(v is None for v in [price, per, pbr, psr, roe]):
                return None

            return {
                'code': code,
                'name': info.get('shortName', code),
                'price': price, 'per': per, 'pbr': pbr, 'psr': psr, 'roe': roe,
                'roe_history': [] # yfinance 과거 데이터 불안정으로 비워둠
            }
        except Exception as e:
            logger.warning(f"{code} (yfinance) 정보 조회 실패: {e}")
            return None

    def _get_stock_financials(self, stock_code: str) -> dict | None:
        """종목 코드에 따라 적절한 핸들러를 사용하여 재무 정보를 가져옵니다."""
        # 국내 주식 (6자리 숫자)
        if len(stock_code) == 6 and stock_code.isdigit():
            # 조회 기준일 (최근 5일 시도)
            for i in range(5):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    result = self._get_financials_pykrx(stock_code, date)
                    # pykrx가 데이터를 성공적으로 반환했는지 확인
                    if result and all(result.get(k) is not None for k in ['price', 'per', 'pbr', 'roe']):
                         # PSR은 yfinance에서 보조적으로 가져오며, 실패해도 일단 진행
                        if result.get('psr') is None:
                            logger.warning(f"{stock_code}: PSR 정보를 yfinance에서 가져오지 못했지만, 나머지 정보로 평가를 진행합니다.")
                        return result
                except Exception:
                    # 특정 날짜에 데이터가 없으면 다음 날짜로 넘어감
                    continue
            return None # 5일간 조회 실패 시 None 반환
        # 해외 주식
        else:
            return self._get_financials_yfinance(stock_code)

    def screen_stocks(self, stock_codes: list[str]) -> pd.DataFrame:
        """주어진 종목 코드 리스트에 대해 스크리닝을 수행하고 별점을 부여합니다."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_code = {executor.submit(self._get_stock_financials, code): code for code in stock_codes}
            for future in concurrent.futures.as_completed(future_to_code):
                try:
                    data = future.result()
                    if data:
                        results.append(self._evaluate_stock(data))
                except Exception as exc:
                    logger.error(f'{future_to_code[future]} 처리 중 예외 발생: {exc}')

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.sort_values(by='stars', ascending=False).reset_index(drop=True)

    def _evaluate_stock(self, data: dict) -> dict:
        """개별 종목에 대해 QVM 규칙을 평가하고 별점을 계산합니다."""
        rules = self.qvm_rules
        conditions_met = 0

        try:
            if data['per'] is not None and data['per'] < rules['per']['max']: conditions_met += 1
            if data['pbr'] is not None and data['pbr'] >= rules['pbr']['min']: conditions_met += 1
            if data['psr'] is not None and data['psr'] <= rules['psr']['max']: conditions_met += 1
            
            # 현재 ROE로만 평가 (10년치 데이터는 비활성화)
            if data['roe'] is not None and data['roe'] >= rules['roe']['min']:
                conditions_met += 1
        except (TypeError, KeyError) as e:
            logger.warning(f"{data.get('code')} 평가 중 오류: {e}. 일부 데이터가 누락되었을 수 있습니다.")


        # 별점 계산
        if conditions_met == 4: stars = 5
        elif conditions_met == 3: stars = 4
        elif conditions_met == 2: stars = 3
        else: stars = conditions_met

        data['stars'] = stars
        data['recommendation'] = "비추천" if stars <= 3 else "추천"
        
        # 데이터 포맷팅
        for key in ['price', 'per', 'pbr', 'psr', 'roe']:
            if isinstance(data.get(key), (int, float)):
                data[key] = f"{data[key]:.2f}"

        return data
