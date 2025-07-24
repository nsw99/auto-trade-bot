import json
import time
from flask import Flask, render_template, Response, jsonify, request
import sys
import os
import pandas as pd
import uuid
import numpy as np
from waitress import serve

# --- 경로 설정: constants.py를 사용하도록 수정 ---
# sys.path에 프로젝트 루트를 추가하여 src 모듈을 임포트할 수 있도록 합니다.
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT_PATH)

from src.constants import (
    PURCHASE_HISTORY_FILE, CONFIG_FILE, QVM_RULES_FILE
)
from src.redis_handler import redis_handler
from src.logger import logger
from src.kis_api import KISApiHandler
from src.config_loader import ConfigLoader
from src.stock_screener import StockScreener, get_kosdaq100_codes, get_nasdaq100_codes
from decimal import Decimal, InvalidOperation


# --- JSON 직렬화 오류 해결을 위한 클래스 ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)


app = Flask(__name__)


# --- 커스텀 JSON 응답 생성 헬퍼 함수 ---
def create_json_response(data, status_code=200):
    json_string = json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
    response = app.response_class(
        response=json_string,
        status=status_code,
        mimetype='application/json'
    )
    return response


# --- 캐시 설정 ---
balance_cache = {"data": None, "timestamp": 0}
quote_cache = {}
screener_cache = {"kosdaq": None, "nasdaq": None, "timestamp": 0}
CACHE_DURATION = 300  # 5분

# --- 핸들러 초기화 ---
try:
    # ConfigLoader 초기화 시 경로를 constants에서 가져와 전달
    config_loader = ConfigLoader(config_path=CONFIG_FILE, qvm_rules_path=QVM_RULES_FILE)
    kis_api_handler = KISApiHandler(config_loader)
    stock_screener = StockScreener(kis_api_handler, config_loader)
except Exception as e:
    logger.error(f"대시보드 핸들러 초기화 실패: {e}")
    kis_api_handler = None
    stock_screener = None


# --- 데이터 로딩 함수들 (내부 코드는 대부분 동일, 경로만 상수로 대체) ---
def run_screener_and_get_results():
    """
    StockScreener를 사용하여 KOSDAQ 및 NASDAQ 종목을 스크리닝하고 결과를 캐시합니다.
    """
    current_time = time.time()
    if current_time - screener_cache.get("timestamp", 0) < CACHE_DURATION:
        logger.info("유효한 스크리너 캐시를 사용합니다.")
        return {
            "kosdaq": screener_cache["kosdaq"],
            "nasdaq": screener_cache["nasdaq"]
        }

    logger.info("스크리너 캐시가 만료되어 실시간 스크리닝을 시작합니다.")
    if not stock_screener:
        logger.error("StockScreener가 초기화되지 않았습니다.")
        return {"kosdaq": [], "nasdaq": []}

    kosdaq_codes = get_kosdaq100_codes()
    nasdaq_codes = get_nasdaq100_codes()

    kosdaq_results_df = stock_screener.screen_stocks(kosdaq_codes)
    nasdaq_results_df = stock_screener.screen_stocks(nasdaq_codes)

    kosdaq_results = kosdaq_results_df.head(10).to_dict('records') if not kosdaq_results_df.empty else []
    nasdaq_results = nasdaq_results_df.head(10).to_dict('records') if not nasdaq_results_df.empty else []

    screener_cache["kosdaq"] = kosdaq_results
    screener_cache["nasdaq"] = nasdaq_results
    screener_cache["timestamp"] = current_time

    return {"kosdaq": kosdaq_results, "nasdaq": nasdaq_results}


def get_grouped_journal_data():
    """
    거래 내역을 '플랜명'과 '종목명'으로 그룹화하고 요약 정보를 계산합니다.
    """
    try:
        # PURCHASE_HISTORY_FILE 상수를 사용
        if not os.path.exists(PURCHASE_HISTORY_FILE) or os.path.getsize(PURCHASE_HISTORY_FILE) == 0:
            return []

        df = pd.read_csv(PURCHASE_HISTORY_FILE)
        if df.empty:
            return []

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

        grouped = df.groupby(['plan_name', 'stock_code', 'stock_name'])

        summary_list = []
        for name, group in grouped:
            plan_name, stock_code, stock_name = name

            total_quantity = group['quantity'].sum()
            total_amount = group['amount'].sum()
            avg_price = total_amount / total_quantity if total_quantity > 0 else 0

            details = []
            for record in group.sort_values(by='timestamp', ascending=False).to_dict('records'):
                cleaned_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}

                cleaned_record['date'] = cleaned_record['timestamp'].strftime('%Y-%m-%d %H:%M')
                if 'id' not in cleaned_record or pd.isna(cleaned_record['id']):
                    cleaned_record['id'] = str(uuid.uuid4())
                details.append(cleaned_record)

            summary = {
                "id": f"group_{plan_name}_{stock_code}",
                "plan_name": plan_name,
                "stock_name": stock_name,
                "stock_code": stock_code,
                "total_quantity": total_quantity,
                "avg_price": f"{avg_price:,.0f}",
                "total_amount": f"{total_amount:,.0f}",
                "trade_count": len(group),
                "last_trade_date": group['timestamp'].max().strftime('%Y-%m-%d'),
                "details": details
            }
            summary_list.append(summary)

        return sorted(summary_list, key=lambda x: x['last_trade_date'], reverse=True)

    except Exception as e:
        logger.error(f"[대시보드] 그룹화된 거래 내역을 읽는 중 오류 발생: {e}", exc_info=True)
        return []


def get_dca_plans_from_file():
    try:
        # CONFIG_FILE 상수를 사용
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"설정 파일({CONFIG_FILE})을 찾을 수 없습니다.")
        return {"dca_plans": []}
    except Exception as e:
        logger.error(f"설정 파일을 읽는 중 오류 발생: {e}")
        return {"dca_plans": []}


def save_dca_plans_to_file(config_data):
    try:
        # CONFIG_FILE 상수를 사용
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"설정 파일을 저장하는 중 오류 발생: {e}")
        return False


def get_summary_chart_data(period):
    """
    기간별 (일별, 주별, 월별) 매매 결산 차트 데이터를 반환합니다.
    """
    try:
        # PURCHASE_HISTORY_FILE 상수를 사용
        if not os.path.exists(PURCHASE_HISTORY_FILE) or os.path.getsize(PURCHASE_HISTORY_FILE) == 0:
            return {"labels": [], "data": []}

        df = pd.read_csv(PURCHASE_HISTORY_FILE)
        if df.empty:
            return {"labels": [], "data": []}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

        df['date_only'] = df['timestamp'].dt.date

        if period == 'daily':
            grouped = df.groupby('date_only')['amount'].sum()
            labels = [d.strftime('%Y-%m-%d') for d in grouped.index]
        elif period == 'weekly':
            df['week'] = df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
            grouped = df.groupby('week')['amount'].sum()
            labels = grouped.index.tolist()
        elif period == 'monthly':
            df['month'] = df['timestamp'].dt.to_period('M').apply(lambda r: r.start_time.strftime('%Y-%m'))
            grouped = df.groupby('month')['amount'].sum()
            labels = grouped.index.tolist()
        else:
            return {"labels": [], "data": []}

        data = grouped.values.tolist()

        return {"labels": labels, "data": data}

    except Exception as e:
        logger.error(f"[대시보드] 요약 차트 데이터 생성 중 오류 발생 ({period}): {e}", exc_info=True)
        return {"labels": [], "data": []}

# --- Flask 라우트들 (이하 변경 없음) ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """차트를 제외한 주요 데이터를 반환합니다."""
    kpis = {}
    if redis_handler:
        raw_kpis = redis_handler.get_kpis()
        for key, value in raw_kpis.items():
            try:
                if key == 'last_updated':
                    kpis[key] = value
                    continue
                decimal_value = Decimal(value)
                if key == 'pnl_rate':
                    kpis[key] = f"{decimal_value:.2f}"
                else:
                    kpis[key] = f"{decimal_value:,.0f}"
            except (ValueError, TypeError, InvalidOperation):
                kpis[key] = value

    trade_journal = get_grouped_journal_data()
    config_data = get_dca_plans_from_file()
    active_plans = [plan for plan in config_data.get('dca_plans', []) if plan.get('enabled', False)]

    qvm_screening_results = run_screener_and_get_results()

    return create_json_response({
        "kpis": kpis,
        "trade_journal": trade_journal,
        "dca_plans": active_plans,
        "qvm_screening_results": qvm_screening_results
    })


@app.route('/api/stock-price/<symbol>')
def get_stock_price(symbol):
    """단일 종목의 현재 시세를 반환합니다."""
    if not kis_api_handler:
        return create_json_response({"error": "KIS API 핸들러가 초기화되지 않았습니다."}, status_code=500)
    try:
        # 캐시 확인
        current_time = time.time()
        if symbol in quote_cache and current_time - quote_cache[symbol]["timestamp"] < CACHE_DURATION:
            logger.info(f"유효한 시세 캐시를 사용합니다: {symbol}")
            return create_json_response(quote_cache[symbol]["data"])

        stock = kis_api_handler.kis.stock(symbol)
        quote = stock.quote()

        # KisQuote 객체를 딕셔너리로 변환
        quote_data = {
            "symbol": quote.symbol,
            "name": quote.name,
            "price": str(quote.price),  # Decimal 타입을 문자열로 변환
            "change": str(quote.change),
            "change_rate": str(quote.change_rate),
            "volume": str(quote.volume),
            "market": quote.market,
            "currency": "USD" if quote.market in ["NASDAQ", "NYSE", "AMEX"] else "KRW"
        }

        # 캐시 저장
        quote_cache[symbol] = {"data": [quote_data], "timestamp": current_time}

        return create_json_response([quote_data])
    except Exception as e:
        logger.error(f"[대시보드] 종목 시세 조회 실패 ({symbol}): {e}", exc_info=True)
        return create_json_response({"error": f"시세 조회 실패: {e}"}, status_code=500)


@app.route('/api/search-stock/<searchTerm>')
def search_stock(searchTerm):
    """종목명 또는 종목코드로 국내 주식을 검색합니다."""
    if not kis_api_handler:
        return create_json_response({"error": "KIS API 핸들러가 초기화되지 않았습니다."}, status_code=500)
    try:
        results = kis_api_handler.kis.stock.search(searchTerm)
        # KisStock 객체 리스트를 딕셔너리 리스트로 변환
        search_results = []
        for stock in results:
            search_results.append({
                "symbol": stock.symbol,
                "name": stock.name,
                "market": stock.market
            })
        return create_json_response(search_results)
    except Exception as e:
        logger.error(f"[대시보드] 종목 검색 실패 ({searchTerm}): {e}", exc_info=True)
        return create_json_response({"error": f"종목 검색 실패: {e}"}, status_code=500)


@app.route('/api/chart/<stockCode>')
def get_stock_chart(stockCode):
    """단일 종목의 OHLCV 차트 데이터를 반환합니다."""
    if not kis_api_handler:
        return create_json_response({"error": "KIS API 핸들러가 초기화되지 않았습니다."}, status_code=500)
    try:
        stock = kis_api_handler.kis.stock(stockCode)
        # 최근 1년치 일봉 데이터 요청
        chart_data = stock.chart(period="day", adjust_price=True, count=365)

        # LightweightCharts 형식에 맞게 데이터 변환
        ohlcv_data = []
        for bar in chart_data:
            ohlcv_data.append({
                "time": bar.date.strftime("%Y-%m-%d"),  # 날짜 형식 맞춤
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume)
            })

        # 최신 데이터가 먼저 오도록 정렬되어 있을 수 있으므로, 시간 순으로 정렬
        ohlcv_data.sort(key=lambda x: x["time"])

        return create_json_response(ohlcv_data)
    except Exception as e:
        logger.error(f"[대시보드] 종목 차트 조회 실패 ({stockCode}): {e}", exc_info=True)
        return create_json_response({"error": f"차트 데이터 조회 실패: {e}"}, status_code=500)


@app.route('/api/plans', methods=['POST'])
def add_dca_plan():
    """새로운 DCA 플랜을 추가합니다."""
    if not config_loader:
        return create_json_response({"error": "설정 로더가 초기화되지 않았습니다."}, status_code=500)

    data = request.get_json()
    plan_name = data.get('plan_name')

    if not plan_name:
        return create_json_response({"error": "플랜 이름은 필수입니다."}, status_code=400)

    config_data = get_dca_plans_from_file()
    dca_plans = config_data.get('dca_plans', [])

    # 중복 플랜 이름 확인
    if any(plan['plan_name'] == plan_name for plan in dca_plans):
        return create_json_response({"error": f"'{plan_name}' 플랜이 이미 존재합니다."}, status_code=409)

    # 새 플랜 데이터 구성 (기본값 포함)
    new_plan = {
        "plan_name": plan_name,
        "stock_code": data.get('stock_code'),
        "stock_name": data.get('stock_name'),
        "market": data.get('market', 'KRX'),  # 기본값 KRX
        "monthly_budget": data.get('monthly_budget', 0),
        "buy_amount_per_trade": data.get('buy_amount_per_trade', 0),
        "enabled": data.get('enabled', False),
        "mode": data.get('mode', 'full_auto'),
        "sell_strategy": data.get('sell_strategy',
                                  {"enabled": True, "profit_target_percent_min": 10, "loss_cut_percent": 5})
    }

    dca_plans.append(new_plan)
    config_data['dca_plans'] = dca_plans

    if save_dca_plans_to_file(config_data):
        logger.info(f"새로운 플랜 추가됨: {plan_name}")
        return create_json_response({"message": "플랜이 성공적으로 추가되었습니다."}, status_code=201)
    else:
        return create_json_response({"error": "플랜 저장 중 오류가 발생했습니다."}, status_code=500)


@app.route('/api/plans/<planName>', methods=['PUT'])
def update_dca_plan(planName):
    """기존 DCA 플랜을 수정합니다."""
    if not config_loader:
        return create_json_response({"error": "설정 로더가 초기화되지 않았습니다."}, status_code=500)

    data = request.get_json()
    config_data = get_dca_plans_from_file()
    dca_plans = config_data.get('dca_plans', [])

    found = False
    for i, plan in enumerate(dca_plans):
        if plan['plan_name'] == planName:
            # 기존 플랜 업데이트
            dca_plans[i] = {
                "plan_name": planName,  # 이름은 변경 불가
                "stock_code": data.get('stock_code', plan.get('stock_code')),
                "stock_name": data.get('stock_name', plan.get('stock_name')),
                "market": data.get('market', plan.get('market', 'KRX')),
                "monthly_budget": data.get('monthly_budget', plan.get('monthly_budget')),
                "buy_amount_per_trade": data.get('buy_amount_per_trade', plan.get('buy_amount_per_trade')),
                "enabled": data.get('enabled', plan.get('enabled')),
                "mode": data.get('mode', plan.get('mode')),
                "sell_strategy": data.get('sell_strategy', plan.get('sell_strategy',
                                                                    {"enabled": True, "profit_target_percent_min": 10,
                                                                     "loss_cut_percent": 5}))
            }
            found = True
            break

    if not found:
        return create_json_response({"error": "해당 플랜을 찾을 수 없습니다."}, status_code=404)

    config_data['dca_plans'] = dca_plans
    if save_dca_plans_to_file(config_data):
        logger.info(f"플랜 수정됨: {planName}")
        return create_json_response({"message": "플랜이 성공적으로 수정되었습니다."})
    else:
        return create_json_response({"error": "플랜 저장 중 오류가 발생했습니다."}, status_code=500)


@app.route('/api/plans/<planName>', methods=['DELETE'])
def delete_dca_plan(planName):
    """DCA 플랜을 삭제합니다."""
    if not config_loader:
        return create_json_response({"error": "설정 로더가 초기화되지 않았습니다."}, status_code=500)

    config_data = get_dca_plans_from_file()
    dca_plans = config_data.get('dca_plans', [])

    original_count = len(dca_plans)
    dca_plans = [plan for plan in dca_plans if plan['plan_name'] != planName]

    if len(dca_plans) == original_count:
        return create_json_response({"error": "해당 플랜을 찾을 수 없습니다."}, status_code=404)

    config_data['dca_plans'] = dca_plans
    if save_dca_plans_to_file(config_data):
        logger.info(f"플랜 삭제됨: {planName}")
        return create_json_response({"message": "플랜이 성공적으로 삭제되었습니다."})
    else:
        return create_json_response({"error": "플랜 삭제 중 오류가 발생했습니다."}, status_code=500)


@app.route('/api/summary_chart/<period>')
def get_summary_chart(period):
    """기간별 라인 차트 데이터를 반환합니다."""
    if period not in ['daily', 'weekly', 'monthly']:
        return create_json_response({"error": "Invalid period"}, status_code=400)

    summary_data = get_summary_chart_data(period)
    return create_json_response(summary_data)


@app.route('/stream')
def stream():
    if not redis_handler:
        return Response("Redis에 연결할 수 없습니다.", status=500)

    def event_stream():
        pubsub = redis_handler.r.pubsub()
        pubsub.subscribe('realtime_logs')
        logger.info("대시보드 클라이언트가 실시간 로그 스트림에 연결되었습니다.")
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    yield f"data: {message['data']}\n\n"
        except GeneratorExit:
            logger.info("대시보드 클라이언트의 연결이 끊어졌습니다.")
        finally:
            pubsub.close()

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/journal', methods=['DELETE'])
def delete_journal_entries():
    data = request.get_json()
    ids_to_delete = data.get('ids', [])
    if not ids_to_delete:
        return create_json_response({"error": "삭제할 ID가 제공되지 않았습니다."}, status_code=400)
    try:
        df = pd.read_csv(PURCHASE_HISTORY_FILE)
        if 'id' not in df.columns:
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            df['id'] = df['id'].apply(lambda x: str(uuid.uuid4()) if pd.isna(x) else x)
        all_detail_ids_to_delete = set()
        grouped_data = get_grouped_journal_data()
        for group_id in ids_to_delete:
            for group in grouped_data:
                if group['id'] == group_id:
                    for detail in group['details']:
                        all_detail_ids_to_delete.add(detail['id'])
        original_count = len(df)
        df = df[~df['id'].isin(all_detail_ids_to_delete)]
        new_count = len(df)
        if original_count == new_count:
            return create_json_response({"error": "삭제할 매매 기록을 찾을 수 없습니다."}, status_code=404)
        df.to_csv(PURCHASE_HISTORY_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"{original_count - new_count}개의 매매 기록을 삭제했습니다.")
        redis_handler.publish('realtime_logs', f"{original_count - new_count}개의 매매 기록이 삭제되었습니다.")
    except FileNotFoundError:
        return create_json_response({"error": "매매 기록 파일을 찾을 수 없습니다."}, status_code=404)
    except Exception as e:
        logger.error(f"매매 기록 삭제 중 오류 발생: {e}", exc_info=True)
        return create_json_response({"error": "매매 기록 삭제 중 오류가 발생했습니다."}, status_code=500)
    finally:
        redis_handler.publish('realtime_logs', f"{original_count - new_count}개의 매매 기록이 삭제되었습니다.")
    logger.info(f"{original_count - new_count}개의 매매 기록이 성공적으로 삭제되었습니다.")
    redis_handler.publish('realtime_logs', f"{original_count - new_count}개의 매매 기록이 성공적으로 삭제되었습니다.")
    return create_json_response({"message": f"{original_count - new_count}개의 매매 기록이 성공적으로 삭제되었습니다."})



if __name__ == '__main__':
    # 개발 환경에서는 app.run()을 사용하고, 배포 환경에서는 waitress.serve()를 사용하는 것이 일반적입니다.
    # 여기서는 waitress를 사용하도록 설정합니다.
    try:
        logger.info("Flask 대시보드 서버를 시작합니다 (Waitress).")
        serve(app, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Flask 서버 시작 실패: {e}", exc_info=True)
