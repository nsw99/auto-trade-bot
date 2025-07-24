import logging
import logging.handlers
import os
from datetime import datetime
from src.redis_handler import redis_handler # Redis 핸들러 인스턴스 임포트

# Redis로 로그를 보내는 커스텀 핸들러
class RedisLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        if redis_handler:
            log_entry = self.format(record)
            redis_handler.publish_log(log_entry)

def setup_logger():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # yfinance 및 requests 라이브러리의 DEBUG 로그 비활성화
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING) # requests가 사용하는 urllib3도 포함

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file = os.path.join(log_dir, f"dca_bot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Redis Handler
    redis_log_handler = RedisLogHandler()
    redis_log_handler.setFormatter(log_formatter)
    logger.addHandler(redis_log_handler)

    logging.info("로거가 설정되었습니다 (Console, File, Redis).")
    return logger

# Initialize logger
logger = setup_logger()
