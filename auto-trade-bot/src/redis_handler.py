import redis
import json
from decimal import Decimal
import numpy as np
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)

class RedisHandler:
    def __init__(self, host='localhost', port=6379, db=0):
        """
        Redis 서버와의 연결을 초기화합니다.
        연결 실패 시 ConnectionError를 발생시킵니다.
        """
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.r.ping()

    def publish_log(self, message):
        """
        로그 메시지를 'realtime_logs' 채널로 발행합니다.
        """
        self.r.publish('realtime_logs', message)

    def store_trade(self, trade_data):
        """
        거래 내역을 'trade_history' 리스트에 저장합니다.
        """
        self.r.lpush('trade_history', json.dumps(trade_data, cls=CustomJSONEncoder))
        self.r.ltrim('trade_history', 0, 99) # 최근 100개만 유지

    def update_kpis(self, kpi_data):
        """
        주요 성과 지표(KPI)를 'dashboard_kpis' 해시에 업데이트합니다.
        호환성을 위해 개별 hset 명령을 사용합니다.
        """
        try:
            for key, value in kpi_data.items():
                self.r.hset('dashboard_kpis', key, str(value))
        except Exception as e:
            # 여기서 예외가 발생하면 호출자가 처리해야 합니다.
            # 이 클래스는 로거를 직접 사용하지 않습니다.
            raise e

    def get_kpis(self):
        """
        저장된 KPI 데이터를 가져옵니다.
        """
        return self.r.hgetall('dashboard_kpis')

    def get_trade_history(self, count=20):
        """
        최근 거래 내역을 지정된 수만큼 가져옵니다.
        """
        trades = self.r.lrange('trade_history', 0, count - 1)
        return [json.loads(trade) for trade in trades]

# 싱글턴 인스턴스 생성
# 이 부분은 애플리케이션 시작점에서 로깅과 함께 처리하는 것이 더 안전합니다.
# 우선 logger.py에서 이 인스턴스를 사용하므로 그대로 둡니다.
try:
    redis_handler = RedisHandler()
except redis.exceptions.ConnectionError as e:
    # 여기서 최소한의 콘솔 출력으로 문제를 알립니다.
    print(f"[CRITICAL] Redis 연결 실패: {e}. Redis 서버가 실행 중인지 확인하세요.")
    redis_handler = None
except Exception as e:
    print(f"[CRITICAL] Redis 핸들러 초기화 중 예기치 않은 오류 발생: {e}")
    redis_handler = None
