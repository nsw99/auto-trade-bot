# src/yfinance_api.py

import yfinance as yf
import pandas as pd
from decimal import Decimal

class YFinanceApiHandler:
    """
    yfinance를 사용하여 미국 주식 데이터를 가져오는 핸들러입니다.
    """
    def _get_ticker(self, stock_code: str):
        return yf.Ticker(stock_code)

    def get_qvm_factors(self, stock_code: str) -> dict | None:
        """
        QVM 팩터 계산에 필요한 재무/시세 데이터를 조회합니다.
        KIS API의 quote()와 fetch_ohlcv() 역할을 한 번에 수행합니다.
        """
        ticker = self._get_ticker(stock_code)
        info = ticker.info

        # yfinance는 ROE, PBR을 직접 제공합니다.
        roe = info.get('returnOnEquity')
        pbr = info.get('priceToBook')
        name = info.get('shortName')
        
        # 필수 데이터가 없으면 건너뜁니다.
        if not all([roe, pbr, name]):
            return None

        # 180일치 시세 데이터를 가져옵니다.
        history = ticker.history(period="180d")
        if history.empty or len(history) < 120:
            return None

        return {
            'roe': Decimal(str(roe)) * 100, # 백분율로 변환
            'pbr': Decimal(str(pbr)),
            'name': name,
            'ohlcv': history
        }