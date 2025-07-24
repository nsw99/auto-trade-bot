import json
from src.logger import logger

class ConfigLoader:
    """
    config.json 파일을 읽고, 유효성을 검사하며,
    설정 데이터에 대한 접근을 제공하는 클래스.
    """
    def __init__(self, config_path='config.json', qvm_rules_path='qvm_rules.json'):
        """
        초기화 메서드. 설정 파일을 불러오고 유효성을 검사합니다.

        Args:
            config_path (str): 설정 파일의 경로.
            qvm_rules_path (str): QVM 규칙 파일의 경로.
        """
        self.config_path = config_path
        self.qvm_rules_path = qvm_rules_path
        self.config = self._load_config()
        self.qvm_rules = self._load_qvm_rules()
        self._validate_config()

    def _load_config(self):
        """설정 파일을 읽어 파이썬 딕셔너리로 변환합니다."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("설정 파일을 성공적으로 불러왔습니다.")
            return config
        except FileNotFoundError:
            logger.error(f"설정 파일({self.config_path})을 찾을 수 없습니다.")
            raise
        except json.JSONDecodeError:
            logger.error(f"설정 파일({self.config_path})의 형식이 올바르지 않습니다.")
            raise

    def _load_qvm_rules(self):
        """QVM 규칙 파일을 읽어 파이썬 딕셔너리로 변환합니다."""
        try:
            with open(self.qvm_rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            logger.info("QVM 규칙 파일을 성공적으로 불러왔습니다.")
            return rules
        except FileNotFoundError:
            logger.warning(f"QVM 규칙 파일({self.qvm_rules_path})을 찾을 수 없습니다. 기본값으로 진행합니다.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"QVM 규칙 파일({self.qvm_rules_path})의 형식이 올바르지 않습니다.")
            return {}

    def _validate_config(self):
        """설정 파일의 필수 키 존재 여부 등 기본 유효성을 검사합니다."""
        logger.info("설정 유효성 검사를 시작합니다...")
        if 'dca_plans' not in self.config or not isinstance(self.config['dca_plans'], list):
            raise ValueError("'dca_plans' 항목이 설정 파일에 없거나 리스트 형식이 아닙니다.")
        logger.info("설정 유효성 검사가 완료되었습니다.")

    def get_config(self):
        """전체 설정 객체를 반환합니다."""
        return self.config

    def get_qvm_rules(self):
        """QVM 규칙 객체를 반환합니다."""
        return self.qvm_rules

    def get_active_plans(self):
        """활성화된(enabled: true) 모든 투자 계획 목록을 반환합니다."""
        return [
            plan for plan in self.config.get('dca_plans', [])
            if plan.get('enabled', False)
        ]
