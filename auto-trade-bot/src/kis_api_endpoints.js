/**
 * 한국투자증권 오픈 API 엔드포인트 전체 목록
 * 문서 버전: 20250714_030000
 * 이 파일은 API의 구조를 파악하고 참고하기 위한 용도입니다.
 */

const KIS_API_ENDPOINTS = {
    // 인증 (Authentication)
    auth: {
        getToken: {
            method: 'POST',
            url: '/oauth2/tokenP',
            description: '접근 토큰 발급 (OAuth 2.0)'
        },
        revokeToken: {
            method: 'POST',
            url: '/oauth2/revokeP',
            description: '접근 토큰 폐기'
        },
        getApprovalKey: {
            method: 'POST',
            url: '/oauth2/Approval',
            description: '웹소켓 접속용 승인 키 발급'
        }
    },

    // 국내주식 (Domestic Stock)
    domesticStock: {
        // 주문
        orderCash: {
            method: 'POST',
            url: '/uapi/domestic-stock/v1/trading/order-cash',
            description: '주식 현금 매수/매도 주문'
        },
        orderCredit: {
            method: 'POST',
            url: '/uapi/domestic-stock/v1/trading/order-credit',
            description: '주식 신용 매수/매도 주문'
        },
        orderReviseCancel: {
            method: 'POST',
            url: '/uapi/domestic-stock/v1/trading/order-rvsecncl',
            description: '주식 정정/취소 주문'
        },
        // 계좌
        inquireBalance: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/trading/inquire-balance',
            description: '주식 잔고 조회'
        },
        inquireDailyCcld: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/trading/inquire-daily-ccld',
            description: '주식 일별 주문 체결 조회'
        },
        inquirePsblOrder: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/trading/inquire-psbl-order',
            description: '매수/매도 가능 수량 조회'
        },
        // 시세
        inquirePrice: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/inquire-price',
            description: '주식 현재가 시세'
        },
        inquireDailyPrice: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/inquire-daily-price',
            description: '주식일봉/주봉/월봉 조회'
        },
        inquireInvestor: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/inquire-investor',
            description: '투자자별 매매 동향 조회'
        },
        inquireMember: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/inquire-member',
            description: '회원사별 매매 동향 조회'
        },
        inquireElwPrice: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/inquire-elw-price',
            description: 'ELW 현재가 시세'
        },
        // 기타
        searchStockInfo: {
            method: 'GET',
            url: '/uapi/domestic-stock/v1/quotations/search-stock-info',
            description: '종목 정보 조회'
        }
    },

    // 해외주식 (Overseas Stock)
    overseasStock: {
        // 주문
        order: {
            method: 'POST',
            url: '/uapi/overseas-stock/v1/trading/order',
            description: '해외주식 주문'
        },
        orderRevise: {
            method: 'POST',
            url: '/uapi/overseas-stock/v1/trading/order-rvsecncl',
            description: '해외주식 정정/취소 주문'
        },
        // 계좌
        inquireBalance: {
            method: 'GET',
            url: '/uapi/overseas-stock/v1/trading/inquire-balance',
            description: '해외주식 잔고 조회'
        },
        inquireConclusion: {
            method: 'GET',
            url: '/uapi/overseas-stock/v1/trading/inquire-ccn',
            description: '해외주식 체결내역 조회'
        },
        // 시세
        inquirePrice: {
            method: 'GET',
            url: '/uapi/overseas-stock/v1/quotations/inquire-price',
            description: '해외주식 현재가'
        },
        inquireDailyPrice: {
            method: 'GET',
            url: '/uapi/overseas-stock/v1/quotations/inquire-daily-price',
            description: '해외주식 기간별 시세'
        }
    },
    
    // 선물옵션 (Derivatives)
    derivatives: {
        // 주문
        order: {
            method: 'POST',
            url: '/uapi/futuresoptions/v1/trading/order',
            description: '선물/옵션 주문'
        },
        // 계좌
        inquireBalance: {
            method: 'GET',
            url: '/uapi/futuresoptions/v1/trading/inquire-balance',
            description: '선물/옵션 잔고 조회'
        },
        // 시세
        inquirePrice: {
            method: 'GET',
            url: '/uapi/futuresoptions/v1/quotations/inquire-price',
            description: '선물/옵션 현재가'
        }
    },

    // 웹소켓 (WebSocket)
    websocket: {
        domesticStock: {
            protocol: 'ws',
            url: 'ws://ops.koreainvestment.com:21000',
            description: '국내주식 실시간 시세/체결'
        },
        overseasStock: {
            protocol: 'ws',
            url: 'ws://ops.koreainvestment.com:31000',
            description: '해외주식 실시간 시세/체결'
        },
        derivatives: {
            protocol: 'ws',
            url: 'ws://ops.koreainvestment.com:21000',
            description: '선물/옵션 실시간 시세/체결'
        }
    }
};

export default KIS_API_ENDPOINTS;
