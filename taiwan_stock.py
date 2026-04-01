import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 網頁基礎配置
st.set_page_config(page_title="台股短線全能分析助手", layout="wide")
st.title("📈 台股短線量價診斷 + 自動掃描系統")

# 2. 定義掃描名單
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
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def safe_metric_text(val, fmt="{:.2f}", suffix=""):
    if pd.isna(val):
        return "--"
    try:
        return fmt.format(val) + suffix
    except Exception:
        return "--"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(full_id, period):
    """抓取股價資料"""
    try:
        data = yf.download(
            full_id,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False
        )
        if data is None or data.empty:
            return pd.DataFrame()
        data = clean_data(data)
        return data
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def get_fundamental_info(full_id):
    try:
        ticker = yf.Ticker(full_id)
        info = ticker.info if ticker.info else {}
        return info
    except Exception:
        return {}


def calc_fundamental_score(stock_id):
    full_id = f"{stock_id}.TW"
    score = 0
    detail = []

    try:
        info = get_fundamental_info(full_id)

        revenue_growth = safe_float(info.get("revenueGrowth"), default=np.nan)
        roe = safe_float(info.get("returnOnEquity"), default=np.nan)
        pe = safe_float(info.get("trailingPE"), default=np.nan)

        # 成長
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

        # 品質
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

        # 估值
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


def prepare_indicator_data(data):
    """
    計算均線後，建立可判讀資料表
    只保留技術分析必要欄位都有值的列，避免最後一筆不完整導致誤判
    """
    df = data.copy()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['VMA5'] = df['Volume'].rolling(5).mean()

    valid_df = df.dropna(subset=['Close', 'MA5', 'MA20', 'Volume', 'VMA5']).copy()
    return df, valid_df


def get_tech_signal(curr, prev):
    """
    技術面判讀：
    - 黃金交叉 / 死亡交叉
    - 均線位置
    - 均線方向
    - 價格是否站上月線
    - 量價是否配合
    """
    price = safe_float(curr['Close'])
    prev_close = safe_float(prev['Close'])
    ma5 = safe_float(curr['MA5'])
    prev_ma5 = safe_float(prev['MA5'])
    ma20 = safe_float(curr['MA20'])
    prev_ma20 = safe_float(prev['MA20'])
    vol = safe_float(curr['Volume'])
    vma5 = safe_float(curr['VMA5'])

    values = [price, prev_close, ma5, prev_ma5, ma20, prev_ma20, vol, vma5]
    if sum(not pd.isna(x) for x in values) < 7:
        return {
            "trend": "資料不足",
            "message": "技術資料不足，暫時無法判讀",
            "level": "中性"
        }

    ma5_up = ma5 > prev_ma5
    ma20_up = ma20 > prev_ma20
    above_ma20 = price > ma20
    above_ma5 = price > ma5

    golden_cross = (ma5 > ma20) and (prev_ma5 <= prev_ma20)
    death_cross = (ma5 < ma20) and (prev_ma5 >= prev_ma20)

    vol_strong = (not pd.isna(vma5)) and vma5 != 0 and vol > vma5 * 1.2

    # 1. 最強訊號：黃金交叉 + 站上月線
    if golden_cross and above_ma20:
        if vol_strong and price > prev_close:
            return {
                "trend": "多頭",
                "message": "🚀 【黃金交叉＋量價配合】短線轉強，可觀察買點。",
                "level": "強"
            }
        return {
            "trend": "多頭",
            "message": "✅ 【黃金交叉】均線翻多，但量能仍需追蹤。",
            "level": "中"
        }

    # 2. 死亡交叉
    if death_cross:
        return {
            "trend": "弱勢",
            "message": "📉 【死亡交叉】5MA 跌破 20MA，短線轉弱，暫不宜買進。",
            "level": "弱"
        }

    # 3. 均線空頭排列
    if (ma5 < ma20) and (not ma5_up) and (not above_ma20):
        return {
            "trend": "弱勢",
            "message": "⚠️ 【均線下彎】5MA 在 20MA 下方且價格位於月線下，偏空看待，不建議買進。",
            "level": "弱"
        }

    # 4. 月線之上但動能普通
    if above_ma20 and ma5 >= ma20:
        if ma5_up and ma20_up:
            return {
                "trend": "多頭",
                "message": "✅ 【均線偏多】站上月線，5MA 與 20MA 皆上行，可續觀察。",
                "level": "中"
            }
        return {
            "trend": "中性偏多",
            "message": "🟡 【月線之上】結構尚可，但均線斜率不夠強，先追蹤。",
            "level": "中"
        }

    # 5. 月線附近整理
    if abs(price - ma20) / ma20 < 0.02:
        return {
            "trend": "中性",
            "message": "🟡 【月線附近整理】尚未明確表態，等待方向確認。",
            "level": "中"
        }

    # 6. 跌破月線
    return {
        "trend": "弱勢",
        "message": "📉 【跌破月線】短線偏弱，建議保守觀望。",
        "level": "弱"
    }


def get_combined_advice(tech_signal, fund_result):
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
            return ("⚠️【基本面尚可，但技術面明顯轉弱】暫不宜買進，等重新站回月線再看。", "warning")
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
    period = st.sidebar.selectbox("觀測區間", ["6mo", "3mo", "1y"], index=0)

    if stock_id:
        full_id = f"{stock_id}.TW"
        raw_data = fetch_stock_data(full_id, period)

        if not raw_data.empty:
            plot_df, valid_df = prepare_indicator_data(raw_data)

            if len(valid_df) < 2:
                st.warning("可用技術資料不足，建議換股或稍後再試。")
            else:
                curr = valid_df.iloc[-1]
                prev = valid_df.iloc[-2]

                price = safe_float(curr['Close'])
                ma5 = safe_float(curr['MA5'])
                ma20 = safe_float(curr['MA20'])
                vol = safe_float(curr['Volume'])
                vma5 = safe_float(curr['VMA5'])

                vol_ratio = np.nan
                bias_5ma = np.nan

                if not pd.isna(vol) and not pd.isna(vma5) and vma5 != 0:
                    vol_ratio = vol / vma5
                if not pd.isna(price) and not pd.isna(ma5) and ma5 != 0:
                    bias_5ma = (price - ma5) / ma5 * 100

                tech_signal = get_tech_signal(curr, prev)
                fund_result = calc_fundamental_score(stock_id)
                combined_msg, combined_type = get_combined_advice(tech_signal, fund_result)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("最新股價", safe_metric_text(price))
                c2.metric(
                    "短線趨勢",
                    "🔥 多頭" if tech_signal["trend"] == "多頭"
                    else ("❄️ 弱勢" if tech_signal["trend"] == "弱勢" else "🟡 中性")
                )
                c3.metric("成交量比", safe_metric_text(vol_ratio, "{:.2f}", "x"))
                c4.metric("5MA 乖離", safe_metric_text(bias_5ma, "{:.1f}", "%"))
                c5.metric("基本面評等", f"{fund_result['level']} ({fund_result['score']})")

                st.subheader("💡 綜合判讀建議")
                render_message_box(combined_msg, combined_type)

                st.subheader("📉 技術面診斷")
                tech_box_type = "success" if tech_signal["level"] == "強" else ("error" if tech_signal["level"] == "弱" else "info")
                render_message_box(tech_signal["message"], tech_box_type)

                st.subheader("🏢 基本面簡評")
                st.write(" / ".join(fund_result["detail"]))

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
                        x=plot_df.index,
                        open=plot_df['Open'],
                        high=plot_df['High'],
                        low=plot_df['Low'],
                        close=plot_df['Close'],
                        name='K線'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=plot_df.index,
                        y=plot_df['MA5'],
                        line=dict(color='orange', width=1.5),
                        name='5MA'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=plot_df.index,
                        y=plot_df['MA20'],
                        line=dict(color='blue', width=1.5),
                        name='20MA'
                    ),
                    row=1, col=1
                )

                bar_colors = [
                    'red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green'
                    for i in range(len(plot_df))
                ]

                fig.add_trace(
                    go.Bar(
                        x=plot_df.index,
                        y=plot_df['Volume'],
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
            st.error("查無資料，請確認代碼，或稍後再試。")


# ---------- 模式二：全自動掃描器 ----------

else:
    st.header("🎯 今日短線機會掃描")
    st.write(f"正在監測：{len(STOCK_LIST)} 檔高活躍股票")

    if st.button("開始掃描市場（自動偵測回踩與轉強）"):
        results = []
        progress_bar = st.progress(0)

        for i, sid in enumerate(STOCK_LIST):
            try:
                raw_data = fetch_stock_data(f"{sid}.TW", "3mo")
                if raw_data.empty:
                    progress_bar.progress((i + 1) / len(STOCK_LIST))
                    continue

                plot_df, valid_df = prepare_indicator_data(raw_data)
                if len(valid_df) < 3:
                    progress_bar.progress((i + 1) / len(STOCK_LIST))
                    continue

                curr = valid_df.iloc[-1]
                prev = valid_df.iloc[-2]
                pprev = valid_df.iloc[-3]

                price = safe_float(curr['Close'])
                ma5 = safe_float(curr['MA5'])
                prev_ma5 = safe_float(prev['MA5'])
                pprev_ma5 = safe_float(pprev['MA5'])
                ma20 = safe_float(curr['MA20'])

                on_support = False
                if not pd.isna(price) and not pd.isna(ma20) and ma20 != 0:
                    on_support = (abs(price - ma20) / ma20 < 0.015) and price >= ma20

                ma5_turn_up = False
                if not pd.isna(ma5) and not pd.isna(prev_ma5) and not pd.isna(pprev_ma5):
                    ma5_turn_up = (ma5 > prev_ma5) and (prev_ma5 <= pprev_ma5)

                status = []
                if on_support:
                    status.append("🎯 回踩月線")
                if ma5_turn_up:
                    status.append("🚀 5MA 剛轉強")

                if status:
                    fund_result = calc_fundamental_score(sid)
                    bias_5ma = (price - ma5) / ma5 * 100 if (not pd.isna(price) and not pd.isna(ma5) and ma5 != 0) else np.nan

                    if fund_result["score"] >= 70:
                        rating = "可觀察"
                    elif fund_result["score"] >= 40:
                        rating = "中性追蹤"
                    else:
                        rating = "偏短線"

                    results.append({
                        "代碼": sid,
                        "價格": safe_metric_text(price),
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