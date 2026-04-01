import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 網頁基礎配置
st.set_page_config(page_title="台股短線全能分析助手", layout="wide")
st.title("📈 台股短線量價診斷 + 自動掃描系統")

# 2. 定義掃描名單 (0050 部分權值股範例)
STOCK_LIST = [
    # 🟦 半導體 / AI 核心
    '2330','2317','2454','2308','2303','3711','3034','2379','2327','6531',
    '3661','3443','3035','2408','2345','6415','4966','6526','5269','8299',

    # 🟩 AI / 伺服器 / 網通
    '2382','6669','3231','3017','2356','3706','4938','4904','5388','2344',
    '6285','3596','8028','2368','4935','8112','4919','2498','6282','3533',

    # 🟥 電子中型成長股
    '3008','3374','5483','6274','3264','3545','8086','6213','6147','6278',
    '4961','4977','3029','8046','5285','6414','6182','4979','5351','3227',

    # 🟨 傳產 / 航運 / 鋼鐵
    '2603','2609','2615','2002','2027','2207','2201','1519','1504','9945',
    '9904','9910','9933','5522','2542','2520','1319','1301','1303','1326',

    # 🟪 金融（穩定參考）
    '2881','2882','2884','2886','2891','2892','5880','2885','2880','5871',

    # 🟧 高股息 / 防禦
    '1216','1101','1102','2912','5876','6505','9917','8926','8464','2105',

    # 🟫 其他活躍股（容易出訊號）
    '3481','8150','6166','3680','4968','6446','9802','2618','2637','2731'
]

# 3. 側邊欄模式切換
st.sidebar.header("功能選單")
mode = st.sidebar.radio("選擇模式", ["單股詳細診斷", "全自動掃描器"])


# ---------- 共用工具函式 ----------

def clean_data(df):
    """整理 yfinance 回傳欄位"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_float(val, default=np.nan):
    """安全轉 float，避免 nan / 型別錯誤"""
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def safe_metric_text(val, fmt="{:.2f}", suffix=""):
    """避免 metric 顯示 nan"""
    if pd.isna(val):
        return "--"
    try:
        return fmt.format(val) + suffix
    except Exception:
        return "--"


def calc_fundamental_score(stock_id):
    """
    簡化版基本面分數
    使用 yfinance info 可取得的欄位：
    - revenueGrowth
    - returnOnEquity
    - trailingPE
    """
    full_id = f"{stock_id}.TW"
    score = 0
    detail = []

    try:
        ticker = yf.Ticker(full_id)
        info = ticker.info if ticker.info else {}

        revenue_growth = safe_float(info.get("revenueGrowth"), default=np.nan)
        roe = safe_float(info.get("returnOnEquity"), default=np.nan)
        pe = safe_float(info.get("trailingPE"), default=np.nan)

        # 成長性：營收成長
        if not pd.isna(revenue_growth):
            if revenue_growth > 0.20:
                score += 35
                detail.append("營收成長強")
            elif revenue_growth > 0.10:
                score += 25
                detail.append("營收成長佳")
            elif revenue_growth > 0:
                score += 10
                detail.append("營收小幅成長")
            else:
                detail.append("營收成長偏弱")
        else:
            detail.append("營收資料不足")

        # 品質：ROE
        if not pd.isna(roe):
            if roe > 0.20:
                score += 35
                detail.append("ROE 優秀")
            elif roe > 0.10:
                score += 25
                detail.append("ROE 尚可")
            elif roe > 0:
                score += 10
                detail.append("ROE 普通")
            else:
                detail.append("ROE 偏弱")
        else:
            detail.append("ROE 資料不足")

        # 估值：PE
        if not pd.isna(pe):
            if 0 < pe < 15:
                score += 25
                detail.append("本益比合理")
            elif 15 <= pe < 25:
                score += 15
                detail.append("本益比中性")
            elif 25 <= pe < 35:
                score += 5
                detail.append("本益比偏高")
            else:
                detail.append("本益比壓力較高")
        else:
            detail.append("本益比資料不足")

        score = min(score, 100)

        if score >= 70:
            level = "強"
        elif score >= 40:
            level = "中"
        else:
            level = "弱"

        return {
            "score": score,
            "level": level,
            "detail": detail
        }

    except Exception:
        return {
            "score": 0,
            "level": "弱",
            "detail": ["基本面資料抓取失敗"]
        }


def get_tech_signal(price, prev_close, ma5, ma20, vol, vma5):
    """技術面簡化判讀"""
    if any(pd.isna(x) for x in [price, prev_close, ma5, ma20, vol, vma5]) or vma5 == 0:
        return {
            "trend": "資料不足",
            "message": "技術資料不足，暫時無法判讀",
            "level": "中性"
        }

    if price > ma20:
        if price > prev_close and vol > vma5 * 1.2:
            return {
                "trend": "多頭",
                "message": "🚀 【量價齊揚】攻擊力道強，適合續抱或觀察強勢延續。",
                "level": "強"
            }
        elif abs(price - ma20) / ma20 < 0.015:
            return {
                "trend": "多頭",
                "message": "🎯 【回測支撐】貼近月線，觀察量縮守穩買點。",
                "level": "中"
            }
        else:
            return {
                "trend": "多頭",
                "message": "✅ 趨勢偏多，守住 5MA 可續觀察。",
                "level": "中"
            }
    else:
        return {
            "trend": "弱勢",
            "message": "📉 【空頭格局】股價在月線下，建議保守觀望。",
            "level": "弱"
        }


def get_combined_advice(tech_signal, fund_result):
    """綜合技術面與基本面輸出人話判讀"""
    tech_level = tech_signal["level"]
    fund_score = fund_result["score"]

    if tech_level == "強":
        if fund_score >= 70:
            return ("🚀【趨勢＋基本面同步】可觀察買點，偏多看待。", "success")
        elif fund_score >= 40:
            return ("✅【趨勢偏強、基本面中性】可觀察，但不宜過度追價。", "info")
        else:
            return ("⚠️【技術強但基本面弱】偏短線題材，不宜重押。", "warning")

    if tech_level == "中":
        if fund_score >= 70:
            return ("👀【基本面穩健、技術面中性】可列觀察名單，等更明確轉強。", "info")
        elif fund_score >= 40:
            return ("🟡【中性整理】可持續追蹤，等待突破或回測確認。", "info")
        else:
            return ("⚠️【技術普通且基本面偏弱】暫時先觀望較佳。", "warning")

    if tech_level == "弱":
        if fund_score >= 70:
            return ("👀【基本面不錯但趨勢偏弱】可以等待止跌或重新站回月線。", "info")
        elif fund_score >= 40:
            return ("⚠️【基本面中性但技術走弱】建議先保守，不急著進場。", "warning")
        else:
            return ("📉【雙弱】技術面與基本面皆弱，建議避開。", "error")

    return ("資料不足，暫時無法完整判讀。", "warning")


def render_message_box(message, box_type):
    if box_type == "success":
        st.success(message)
    elif box_type == "info":
        st.info(message)
    elif box_type == "warning":
        st.warning(message)
    elif box_type == "error":
        st.error(message)
    else:
        st.write(message)


# ---------- 模式一：單股詳細診斷 ----------

if mode == "單股詳細診斷":
    st.sidebar.markdown("---")
    stock_id = st.sidebar.text_input("輸入台股代碼", value="2330")
    period = st.sidebar.selectbox("觀測區間", ["3mo", "6mo", "1y"], index=0)

    if stock_id:
        full_id = f"{stock_id}.TW"
        data = yf.download(full_id, period=period, progress=False)

        if not data.empty:
            data = clean_data(data)

            data['MA5'] = data['Close'].rolling(5).mean()
            data['MA20'] = data['Close'].rolling(20).mean()
            data['VMA5'] = data['Volume'].rolling(5).mean()

            if len(data) < 20:
                st.warning("資料不足 20 日，建議擴大觀測區間。")
            else:
                curr = data.iloc[-1]
                prev = data.iloc[-2]

                price = safe_float(curr['Close'])
                prev_close = safe_float(prev['Close'])
                ma5 = safe_float(curr['MA5'])
                ma20 = safe_float(curr['MA20'])
                vol = safe_float(curr['Volume'])
                vma5 = safe_float(curr['VMA5'])

                # 避免除以 0 / nan
                vol_ratio = np.nan
                bias_5ma = np.nan
                if not pd.isna(vol) and not pd.isna(vma5) and vma5 != 0:
                    vol_ratio = vol / vma5
                if not pd.isna(price) and not pd.isna(ma5) and ma5 != 0:
                    bias_5ma = (price - ma5) / ma5 * 100

                # 判讀
                tech_signal = get_tech_signal(price, prev_close, ma5, ma20, vol, vma5)
                fund_result = calc_fundamental_score(stock_id)
                combined_msg, combined_type = get_combined_advice(tech_signal, fund_result)

                # 頂部指標
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("最新股價", safe_metric_text(price))
                c2.metric("短線趨勢", f"🔥 多頭" if tech_signal["trend"] == "多頭" else ("❄️ 弱勢" if tech_signal["trend"] == "弱勢" else "—"))
                c3.metric("成交量比", safe_metric_text(vol_ratio, "{:.2f}", "x"))
                c4.metric("5MA 乖離", safe_metric_text(bias_5ma, "{:.1f}", "%"))
                c5.metric("基本面評等", f"{fund_result['level']} ({fund_result['score']})")

                # 綜合建議
                st.subheader("💡 綜合判讀建議")
                render_message_box(combined_msg, combined_type)

                # 技術面說明
                st.subheader("📉 技術面診斷")
                render_message_box(tech_signal["message"], "info" if tech_signal["level"] != "弱" else "error")

                # 基本面說明
                st.subheader("🏢 基本面簡評")
                st.write(" / ".join(fund_result["detail"]))

                # 圖表
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=('K線均線', '成交量'),
                    row_heights=[0.7, 0.3]
                )

                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='K線'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MA5'],
                        line=dict(color='orange', width=1.5),
                        name='5MA'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MA20'],
                        line=dict(color='blue', width=1.5),
                        name='20MA'
                    ),
                    row=1, col=1
                )

                # 台股常見習慣：紅漲綠跌
                bar_colors = [
                    'red' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'green'
                    for i in range(len(data))
                ]

                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        marker_color=bar_colors,
                        name='成交量'
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=650,
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("查無資料，請確認代碼。")


# ---------- 模式二：全自動掃描器 ----------

else:
    st.header("🎯 今日短線機會掃描")
    st.write(f"正在監測：{', '.join(STOCK_LIST)} 等權值股")

    if st.button("開始掃描市場（自動偵測回踩與轉強）"):
        results = []
        progress_bar = st.progress(0)

        for i, sid in enumerate(STOCK_LIST):
            try:
                data = yf.download(f"{sid}.TW", period="1mo", progress=False)

                if not data.empty and len(data) >= 20:
                    data = clean_data(data)
                    data['MA5'] = data['Close'].rolling(5).mean()
                    data['MA20'] = data['Close'].rolling(20).mean()

                    curr = data.iloc[-1]
                    prev = data.iloc[-2]
                    pprev = data.iloc[-3]

                    price = safe_float(curr['Close'])
                    ma5 = safe_float(curr['MA5'])
                    prev_ma5 = safe_float(prev['MA5'])
                    pprev_ma5 = safe_float(pprev['MA5'])
                    ma20 = safe_float(curr['MA20'])

                    if any(pd.isna(x) for x in [price, ma5, prev_ma5, pprev_ma5, ma20]):
                        progress_bar.progress((i + 1) / len(STOCK_LIST))
                        continue

                    # 邏輯：回踩月線 或 5MA 轉強
                    on_support = (abs(price - ma20) / ma20 < 0.015) and price >= ma20 if ma20 != 0 else False
                    ma5_turn_up = (ma5 > prev_ma5) and (prev_ma5 <= pprev_ma5)

                    status = []
                    if on_support:
                        status.append("🎯 回踩月線")
                    if ma5_turn_up:
                        status.append("🚀 5MA 剛轉強")

                    if status:
                        fund_result = calc_fundamental_score(sid)
                        bias_5ma = (price - ma5) / ma5 * 100 if ma5 != 0 else np.nan

                        if fund_result["score"] >= 70:
                            rating = "可觀察"
                        elif fund_result["score"] >= 40:
                            rating = "中性追蹤"
                        else:
                            rating = "偏短線"

                        results.append({
                            "代碼": sid,
                            "價格": f"{price:.2f}",
                            "訊號": " + ".join(status),
                            "基本面": f'{fund_result["level"]} ({fund_result["score"]})',
                            "評價": rating,
                            "5MA乖離": safe_metric_text(bias_5ma, "{:.1f}", "%")
                        })

            except Exception:
                pass

            progress_bar.progress((i + 1) / len(STOCK_LIST))

        if results:
            df_result = pd.DataFrame(results)
            st.dataframe(df_result, use_container_width=True)
            st.balloons()
        else:
            st.info("目前名單中無符合條件的股票，建議繼續等待機會。")

st.caption("免責聲明：本程式僅供技術分析練習與資料視覺化整理，不構成任何投資建議。")