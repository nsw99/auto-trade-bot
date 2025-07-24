import os

# --- 기본 경로 설정 ---
# 이 파일(constants.py)의 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.
# os.path.abspath(__file__) -> /path/to/project/src/constants.py
# os.path.dirname(...) -> /path/to/project/src
# os.path.dirname(...) -> /path/to/project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 주요 폴더 및 파일 경로 ---
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
CONFIG_DIR = os.path.join(PROJECT_ROOT) # config.json이 루트에 있으므로

# 개별 파일 경로
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
QVM_RULES_FILE = os.path.join(CONFIG_DIR, 'qvm_rules.json')
PURCHASE_HISTORY_FILE = os.path.join(LOGS_DIR, 'purchase_history.csv')
SELL_HISTORY_FILE = os.path.join(LOGS_DIR, 'sell_history.csv')

# --- 기타 상수 ---
# 필요한 다른 전역 상수들을 여기에 추가할 수 있습니다.
# 예: CACHE_DURATION = 300
