import argparse
import sys
import os

# sys.path에 프로젝트 루트를 추가하여 src 모듈을 임포트할 수 있도록 합니다.
# start.bat에서 PYTHONPATH를 설정해주지만, 단독 실행을 고려하여 추가합니다.
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT_PATH)

from src.dca_bot import DCABot
from src.logger import logger
# --- 경로 설정: constants.py를 사용하도록 수정 ---
from src.constants import CONFIG_FILE

def main():
    """
    DCA 봇의 메인 진입점.
    봇을 초기화하고 실행합니다.
    """
    parser = argparse.ArgumentParser(description="한국투자증권을 위한 간단한 DCA(분할 매수) 트레이딩 봇입니다.")
    parser.add_argument(
        '--config',
        type=str,
        # CONFIG_FILE 상수를 기본값으로 사용
        default=CONFIG_FILE,
        help='설정 파일의 경로입니다.'
    )

    args = parser.parse_args()

    try:
        logger.info("DCA 봇 애플리케이션을 시작합니다...")
        # DCABot 초기화 시 args.config (명령줄 인자 또는 기본값) 전달
        bot = DCABot(config_path=args.config)
        bot.run()
    except FileNotFoundError as e:
        logger.error(f"필수 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"설정 오류가 발생했습니다: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("봇을 정상적으로 종료합니다...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"예기치 않은 오류가 발생했습니다: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
