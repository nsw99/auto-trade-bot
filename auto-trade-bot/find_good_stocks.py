from src.stock_screener import StockScreener, get_kosdaq100_codes, get_nasdaq100_codes
from src.kis_api import KISApiHandler
from src.config_loader import ConfigLoader
from src.logger import logger
import os
from src.yfinance_api import YFinanceApiHandler
from rich.console import Console
from rich.table import Table
import pandas as pd
# --- 경로 설정: constants.py를 사용하도록 수정 ---
from src.constants import CONFIG_FILE, QVM_RULES_FILE, LOGS_DIR


def main():
    console = Console()
    console.print("[bold green]QVM 주식 스크리너를 시작합니다...[/bold green]")

    try:
        # --- 국내(코스닥) 스크리닝 ---
        console.print("\n[bold] domestic KOSDAQ 100 Screening...[/bold]")
        # CONFIG_FILE, QVM_RULES_FILE 상수를 사용
        config_loader = ConfigLoader(config_path=CONFIG_FILE, qvm_rules_path=QVM_RULES_FILE)
        kis_api_handler = KISApiHandler(config_loader)
        kosdaq_codes = get_kosdaq100_codes()

        # StockScreener 초기화 시 config_loader 전달
        kosdaq_screener = StockScreener(kis_api_handler, config_loader)
        kosdaq_results = kosdaq_screener.screen_stocks(kosdaq_codes)

        # --- 해외(나스닥) 스크리닝 ---
        console.print("\n[bold] overseas NASDAQ 100 Screening...[/bold]")
        # 해외 스크리닝에도 동일한 config_loader 사용
        nasdaq_screener = StockScreener(kis_api_handler, config_loader)
        nasdaq_codes = get_nasdaq100_codes()
        nasdaq_results = nasdaq_screener.screen_stocks(nasdaq_codes)

        # 코스닥 스크리닝 결과를 JSON 파일로 저장
        if not kosdaq_results.empty:
            # LOGS_DIR 상수를 사용
            kosdaq_results_path = os.path.join(LOGS_DIR, 'qvm_kosdaq_results.json')
            kosdaq_results.to_json(kosdaq_results_path, orient='records', indent=4, force_ascii=False)
            console.print(f"[bold green]코스닥 스크리닝 결과가 {kosdaq_results_path}에 저장되었습니다.[/bold green]")

        # 나스닥 스크리닝 결과를 JSON 파일로 저장
        if not nasdaq_results.empty:
            # LOGS_DIR 상수를 사용
            nasdaq_results_path = os.path.join(LOGS_DIR, 'qvm_nasdaq_results.json')
            nasdaq_results.to_json(nasdaq_results_path, orient='records', indent=4, force_ascii=False)
            console.print(f"[bold green]나스닥 스크리닝 결과가 {nasdaq_results_path}에 저장되었습니다.[/bold green]")

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        console.print(f"[bold red]오류 발생: {e}[/bold red]")

if __name__ == "__main__":
    # 더 이상 PROJECT_ROOT를 여기서 정의할 필요 없음
    main()
