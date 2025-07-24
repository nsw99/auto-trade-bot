import os
import csv
from datetime import datetime
from src.logger import logger
# --- 경로 설정: constants.py를 사용하도록 수정 ---
from src.constants import LOGS_DIR, PURCHASE_HISTORY_FILE, SELL_HISTORY_FILE

# CSV 헤더는 그대로 유지
CSV_HEADER = [
    "timestamp", "transaction_type", "plan_name", "stock_code", "stock_name", 
    "quantity", "price", "amount", "reason", "order_id"
]

def _log_transaction(file_path, details: dict, tx_type: str):
    """
    거래 내역(매수/매도)을 지정된 CSV 파일에 기록합니다.
    """
    try:
        # LOGS_DIR 상수를 사용하여 로그 디렉토리 생성
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(CSV_HEADER)

            row = [
                details.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                tx_type,
                details.get('plan_name', ''),
                details.get('stock_code', ''),
                details.get('stock_name', ''),
                details.get('quantity', 0),
                str(details.get('price', '0')), # Decimal을 문자열로 변환
                str(details.get('amount', '0')), # Decimal을 문자열로 변환
                details.get('reason', ''),
                details.get('order_id', '')
            ]
            writer.writerow(row)
        
        logger.info(f"{tx_type.capitalize()} 내역을 {file_path} 파일에 기록했습니다.")

    except Exception as e:
        logger.error(f"{tx_type.capitalize()} 내역을 CSV 파일에 기록하는 중 오류가 발생했습니다: {e}")

def log_purchase(details: dict, tx_type: str = 'buy'):
    """
    거래 내역을 유형에 따라 적절한 파일에 기록합니다.
    """
    if tx_type == 'sell':
        # SELL_HISTORY_FILE 상수를 사용
        _log_transaction(SELL_HISTORY_FILE, details, 'sell')
    else:
        # PURCHASE_HISTORY_FILE 상수를 사용
        _log_transaction(PURCHASE_HISTORY_FILE, details, 'buy')
