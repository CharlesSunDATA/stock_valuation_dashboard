"""
股票估值區間分析儀表板 (Stock Valuation Dashboard)
以 yfinance 抓取資料、兩階段 DCF 估算每股內在價值，並以 Streamlit 互動呈現。
所有邏輯集中於本檔案。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 資料結構：匯總從 yfinance 與 DCF 計算之結果，便於傳遞至 UI
# ---------------------------------------------------------------------------


@dataclass
class MarketData:
    """市場價格與相對估值乘數。"""

    price: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg: float | None
    pb: float | None


@dataclass
class FundamentalInputs:
    """DCF 所需之基本面輸入（已做缺漏處理）。"""

    fcf_base: float  # 最近一期推算之自由現金流（美元）
    shares: float  # 流通在外股數
    net_debt: float  # 淨負債 = 總負債 - 現金（美元）；可為負（淨現金）
    used_defaults: list[str]  # 哪些欄位使用預設值，供 UI 提示


@dataclass
class DCFResult:
    """兩階段 DCF 輸出。"""

    intrinsic_per_share: float
    band_low: float  # 內在價值 -10%
    band_high: float  # 內在價值 +10%
    enterprise_value: float
    pv_fcf_5y: float
    pv_terminal: float


# ---------------------------------------------------------------------------
# yfinance：抓取股價、乘數、現金流與資產負債相關欄位
# ---------------------------------------------------------------------------


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_ticker_info(ticker: str) -> tuple[dict[str, Any], Any]:
    """回傳 (info 字典, Ticker 物件)。失敗時拋出例外由外層處理。"""
    t = yf.Ticker(ticker)
    info = t.info
    if not info:
        raise ValueError(f"無法取得 {ticker} 的基本面資訊（info 為空）。")
    return info, t


def extract_market_data(info: dict[str, Any], hist_close: float | None) -> MarketData:
    """從 info 與可選歷史收盤價組裝相對估值與現價。"""
    price = _safe_float(
        hist_close if hist_close is not None else info.get("currentPrice") or info.get("regularMarketPrice")
    )
    return MarketData(
        price=price,
        trailing_pe=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        peg=_safe_float(info.get("pegRatio")),
        pb=_safe_float(info.get("priceToBook")),
    )


def _latest_annual_value(cf: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """從現金流量表取最近一年度之列（欄位為日期，取第一欄通常為最新年）。"""
    if cf is None or cf.empty:
        return None
    for name in row_names:
        if name in cf.index:
            row = cf.loc[name]
            # 取第一個非空數值（yfinance 年報常為最新期在左）
            for col in cf.columns:
                v = row[col]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
    return None


def _latest_bs_value(bs: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """資產負債表取最近一期。"""
    if bs is None or bs.empty:
        return None
    for name in row_names:
        if name in bs.index:
            row = bs.loc[name]
            for col in bs.columns:
                v = row[col]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
    return None


def build_fundamental_inputs(t: yf.Ticker, info: dict[str, Any]) -> FundamentalInputs:
    """
    推算 FCF = 營運現金流 + 資本支出（CapEx 在報表上多為負值）。
    股數、淨負債優先由 info，失敗則由資產負債表。
    """
    used_defaults: list[str] = []
    cf = None
    bs = None
    try:
        cf = t.cashflow
    except Exception:
        used_defaults.append("現金流量表下載失敗")

    ocf = _latest_annual_value(
        cf if cf is not None else pd.DataFrame(),
        ("Operating Cash Flow", "Cash From Operating Activities", "Cash Flow From Operations"),
    )
    capex = _latest_annual_value(
        cf if cf is not None else pd.DataFrame(),
        ("Capital Expenditure", "Capital Expenditures"),
    )

    if ocf is None:
        ocf = 0.0
        used_defaults.append("營運現金流缺失，FCF 之營運部分以 0 計")
    if capex is None:
        capex = 0.0
        used_defaults.append("資本支出缺失，FCF 之 CapEx 以 0 計")

    # CapEx 通常為負：FCF = OCF + CapEx
    fcf_base = float(ocf) + float(capex)

    shares = _safe_float(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        try:
            bs = t.balance_sheet
            sh_row = _latest_bs_value(bs, ("Ordinary Shares Number", "Share Issued", "Common Stock"))
            if sh_row is not None and sh_row > 0:
                shares = float(sh_row)
            else:
                shares = 1.0
                used_defaults.append("流通在外股數缺失，每股價值分母暫用 1（僅供展示，請勿作實盤）")
        except Exception:
            shares = 1.0
            used_defaults.append("流通在外股數缺失，每股價值分母暫用 1")

    total_debt = _safe_float(info.get("totalDebt"))
    total_cash = _safe_float(info.get("totalCash"))
    if total_debt is None:
        try:
            if bs is None:
                bs = t.balance_sheet
            td = _latest_bs_value(bs, ("Total Debt", "Long Term Debt"))
            total_debt = td if td is not None else 0.0
            if td is None:
                used_defaults.append("總負債由缺省改為 0")
        except Exception:
            total_debt = 0.0
            used_defaults.append("總負債缺失改為 0")
    if total_cash is None:
        try:
            if bs is None:
                bs = t.balance_sheet
            tc = _latest_bs_value(
                bs,
                ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            )
            total_cash = tc if tc is not None else 0.0
            if tc is None:
                used_defaults.append("現金缺失改為 0")
        except Exception:
            total_cash = 0.0
            used_defaults.append("現金缺失改為 0")

    net_debt = float(total_debt) - float(total_cash)

    return FundamentalInputs(
        fcf_base=fcf_base,
        shares=max(float(shares), 1e-12),
        net_debt=net_debt,
        used_defaults=used_defaults,
    )


# ---------------------------------------------------------------------------
# DCF：兩階段（5 年明確成長 + Gordon 成長終值）
# ---------------------------------------------------------------------------


def run_two_stage_dcf(
    fcf0: float,
    growth_5y: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> DCFResult | None:
    """
    第一階段：FCF_t = FCF_{t-1} * (1+g)，t=1..5，各期折現。
    第二階段：TV = FCF_5*(1+g_term)/(WACC-g_term)，折現至今日。
    企業價值 EV = PV(FCF1..5) + PV(TV)；股權價值 = EV - 淨負債；每股 = 股權/股數。
    估值區間：每股內在價值 ±10%。
    """
    if shares <= 0:
        return None
    if wacc <= terminal_growth:
        raise ValueError("WACC 必須大於永續成長率，否則終值公式不成立。")
    if fcf0 <= 0:
        # 仍允許計算但可能不具經濟意義，由 UI 提示
        pass

    pv_fcf = 0.0
    fcf_prev = fcf0
    for t in range(1, 6):
        fcf_t = fcf_prev * (1.0 + growth_5y)
        pv_fcf += fcf_t / ((1.0 + wacc) ** t)
        fcf_prev = fcf_t

    fcf5 = fcf_prev  # 第 5 年終了之 FCF
    tv = fcf5 * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = tv / ((1.0 + wacc) ** 5)

    ev = pv_fcf + pv_terminal
    equity = ev - net_debt
    intrinsic = equity / shares

    band_low = intrinsic * 0.9
    band_high = intrinsic * 1.1

    return DCFResult(
        intrinsic_per_share=intrinsic,
        band_low=band_low,
        band_high=band_high,
        enterprise_value=ev,
        pv_fcf_5y=pv_fcf,
        pv_terminal=pv_terminal,
    )


# ---------------------------------------------------------------------------
# 安全邊際文字與簡易視覺化資料
# ---------------------------------------------------------------------------


def classify_valuation(price: float | None, low: float, high: float) -> str:
    """依當前股價與合理區間 [low, high] 給出繁中標籤。"""
    if price is None or price <= 0:
        return "無法判定（缺少股價）"
    if price < low:
        return "嚴重低估（低於合理區間下緣）"
    if price <= high:
        return "合理（落在估值區間內）"
    return "高估（高於合理區間上緣）"


# ---------------------------------------------------------------------------
# Streamlit 主程式
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="股票估值區間分析", layout="wide")
    st.title("股票估值區間分析儀表板")
    st.caption("資料來源：yfinance｜絕對估值採簡化兩階段 DCF，僅供教育與研究，非投資建議。")

    # --- 側邊欄參數 ---
    with st.sidebar:
        st.header("標的與 DCF 參數")
        ticker = st.text_input("股票代號", value="NVDA").strip().upper() or "NVDA"
        g5 = st.slider("未來 5 年預估 FCF 成長率 (%)", 0, 100, 20, format="%d%%") / 100.0
        g_term = st.slider("永續成長率 Terminal Growth (%)", 1, 5, 3, format="%d%%") / 100.0
        wacc = st.slider("折現率 WACC (%)", 5, 20, 10, format="%d%%") / 100.0

    # --- 抓取資料 ---
    market: MarketData | None = None
    fundamentals: FundamentalInputs | None = None
    err_msg: str | None = None

    try:
        info, t = fetch_ticker_info(ticker)
        hist = t.history(period="5d")
        close_px = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        market = extract_market_data(info, close_px)
        fundamentals = build_fundamental_inputs(t, info)
    except Exception as e:
        err_msg = str(e)
        st.error(f"資料抓取失敗：{err_msg}")
        st.info("請確認代號正確、網路正常，或稍後再試。")
        return

    assert market is not None and fundamentals is not None

    # --- DCF 計算 ---
    dcf: DCFResult | None = None
    dcf_error: str | None = None
    try:
        dcf = run_two_stage_dcf(
            fcf0=fundamentals.fcf_base,
            growth_5y=g5,
            wacc=wacc,
            terminal_growth=g_term,
            net_debt=fundamentals.net_debt,
            shares=fundamentals.shares,
        )
    except ValueError as e:
        dcf_error = str(e)
    except Exception as e:
        dcf_error = f"DCF 計算異常：{e}"

    # --- 區塊一：基本面與相對估值 ---
    st.subheader("一、基本面與相對估值")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("目前股價 (USD)", f"${market.price:,.2f}" if market.price else "—")
    with c2:
        st.metric("Trailing P/E", f"{market.trailing_pe:.2f}" if market.trailing_pe else "—")
    with c3:
        st.metric("Forward P/E", f"{market.forward_pe:.2f}" if market.forward_pe else "—")
    with c4:
        st.metric("PEG Ratio", f"{market.peg:.2f}" if market.peg else "—")
    with c5:
        st.metric("P/B", f"{market.pb:.2f}" if market.pb else "—")

    if fundamentals.used_defaults:
        with st.expander("資料缺漏提示（已使用預設或替代值）", expanded=False):
            for line in fundamentals.used_defaults:
                st.warning(line)

    st.divider()

    # --- 區塊二：DCF 絕對估值 ---
    st.subheader("二、DCF 絕對估值（兩階段）")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
| 項目 | 數值 |
|------|------|
| 基期自由現金流 FCF₀（推算） | {fundamentals.fcf_base:,.0f} USD |
| 流通在外股數 | {fundamentals.shares:,.0f} |
| 淨負債（總負債−現金） | {fundamentals.net_debt:,.0f} USD |
"""
        )
    with col_b:
        st.markdown(
            f"""
| DCF 參數 | 設定 |
|----------|------|
| 5 年 FCF 年化成長 | {g5*100:.1f}% |
| 永續成長率 | {g_term*100:.1f}% |
| WACC | {wacc*100:.1f}% |
"""
        )

    if dcf_error:
        st.error(dcf_error)
    elif dcf is not None:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("每股內在價值（估計）", f"${dcf.intrinsic_per_share:,.2f}")
        with m2:
            st.metric("合理區間下緣 (−10%)", f"${dcf.band_low:,.2f}")
        with m3:
            st.metric("合理區間上緣 (+10%)", f"${dcf.band_high:,.2f}")
        st.caption(
            f"企業價值 EV ≈ {dcf.enterprise_value:,.0f} USD｜"
            f"5 年 FCF 現值 {dcf.pv_fcf_5y:,.0f}｜終值現值 {dcf.pv_terminal:,.0f}"
        )
    else:
        st.warning("無法完成 DCF（請檢查輸入與資料）。")

    st.divider()

    # --- 區塊三：安全邊際 ---
    st.subheader("三、安全邊際與股價對照")
    if dcf is not None and market.price:
        label = classify_valuation(market.price, dcf.band_low, dcf.band_high)
        st.markdown(f"### 結論：**{label}**")

        # 以長條圖對比：區間與現價（單位：美元）
        chart_df = pd.DataFrame(
            {
                "項目": ["合理區間下緣", "目前股價", "每股內在價值", "合理區間上緣"],
                "USD": [dcf.band_low, market.price, dcf.intrinsic_per_share, dcf.band_high],
            }
        ).set_index("項目")
        st.bar_chart(chart_df)

        # 進度條：現價在 [0.8*low, 1.2*high] 線性映射（簡化視覺）
        lo = min(dcf.band_low * 0.85, market.price * 0.5)
        hi = max(dcf.band_high * 1.15, market.price * 1.5)
        if hi > lo:
            pct = (market.price - lo) / (hi - lo)
            pct = max(0.0, min(1.0, pct))
        else:
            pct = 0.5
        st.progress(pct)
        st.caption(f"進度條為股價在參考刻度上的相對位置（僅示意；約 {lo:.1f} ~ {hi:.1f} USD）。")
    else:
        st.info("缺少股價或 DCF 結果，無法顯示安全邊際圖表。")


if __name__ == "__main__":
    main()
