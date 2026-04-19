"""
Stock Valuation Dashboard — single-file Streamlit app.
Fetches data via yfinance, runs a two-stage DCF for intrinsic value per share.
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
# Data containers for yfinance + DCF outputs
# ---------------------------------------------------------------------------


@dataclass
class MarketData:
    """Spot price and relative valuation multiples."""

    price: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg: float | None
    pb: float | None


@dataclass
class FundamentalInputs:
    """Fundamental inputs for DCF (with fallbacks applied)."""

    fcf_base: float  # Implied base-year FCF (USD)
    shares: float  # Shares outstanding
    net_debt: float  # Total debt minus cash (USD); can be negative (net cash)
    used_defaults: list[str]  # Human-readable notes when defaults were used


@dataclass
class DCFResult:
    """Two-stage DCF outputs."""

    intrinsic_per_share: float
    band_low: float  # intrinsic -10%
    band_high: float  # intrinsic +10%
    enterprise_value: float
    pv_fcf_5y: float
    pv_terminal: float


# ---------------------------------------------------------------------------
# yfinance: price, multiples, cash flows, balance sheet
# ---------------------------------------------------------------------------


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_ticker_info(ticker: str) -> tuple[dict[str, Any], Any]:
    """Return (info dict, Ticker). Raises if fundamentals are unavailable."""
    t = yf.Ticker(ticker)
    info = t.info
    if not info:
        raise ValueError(f"Could not load fundamentals for {ticker} (empty info).")
    return info, t


def extract_market_data(info: dict[str, Any], hist_close: float | None) -> MarketData:
    """Build relative multiples and price from info and optional last close."""
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
    """Latest annual value from cash flow statement (iterate columns for first non-null)."""
    if cf is None or cf.empty:
        return None
    for name in row_names:
        if name in cf.index:
            row = cf.loc[name]
            for col in cf.columns:
                v = row[col]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
    return None


def _latest_bs_value(bs: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """Latest value from balance sheet."""
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
    FCF = Operating cash flow + Capital expenditure (CapEx is usually negative).
    Shares and net debt: prefer info, else balance sheet.
    """
    used_defaults: list[str] = []
    cf = None
    bs = None
    try:
        cf = t.cashflow
    except Exception:
        used_defaults.append("Cash flow statement download failed")

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
        used_defaults.append("Operating cash flow missing; OCF in FCF set to 0")
    if capex is None:
        capex = 0.0
        used_defaults.append("CapEx missing; CapEx in FCF set to 0")

    # CapEx is usually negative: FCF = OCF + CapEx
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
                used_defaults.append(
                    "Shares outstanding missing; denominator set to 1 (illustrative only, not for trading)"
                )
        except Exception:
            shares = 1.0
            used_defaults.append("Shares outstanding missing; denominator set to 1")

    total_debt = _safe_float(info.get("totalDebt"))
    total_cash = _safe_float(info.get("totalCash"))
    if total_debt is None:
        try:
            if bs is None:
                bs = t.balance_sheet
            td = _latest_bs_value(bs, ("Total Debt", "Long Term Debt"))
            total_debt = td if td is not None else 0.0
            if td is None:
                used_defaults.append("Total debt defaulted to 0")
        except Exception:
            total_debt = 0.0
            used_defaults.append("Total debt missing; set to 0")
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
                used_defaults.append("Cash defaulted to 0")
        except Exception:
            total_cash = 0.0
            used_defaults.append("Cash missing; set to 0")

    net_debt = float(total_debt) - float(total_cash)

    return FundamentalInputs(
        fcf_base=fcf_base,
        shares=max(float(shares), 1e-12),
        net_debt=net_debt,
        used_defaults=used_defaults,
    )


# ---------------------------------------------------------------------------
# DCF: two-stage (5-year explicit growth + Gordon terminal value)
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
    Stage 1: FCF_t = FCF_{t-1} * (1+g), t=1..5, discount each year.
    Stage 2: TV = FCF_5*(1+g_term)/(WACC-g_term), discount to today.
    EV = PV(FCF1..5) + PV(TV); equity = EV - net debt; per share = equity / shares.
    Fair band: intrinsic per share ±10%.
    """
    if shares <= 0:
        return None
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth (terminal value formula requires it).")
    if fcf0 <= 0:
        pass  # still compute; may be economically weak — surfaced in UI

    pv_fcf = 0.0
    fcf_prev = fcf0
    for t in range(1, 6):
        fcf_t = fcf_prev * (1.0 + growth_5y)
        pv_fcf += fcf_t / ((1.0 + wacc) ** t)
        fcf_prev = fcf_t

    fcf5 = fcf_prev
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
# Valuation label + chart helpers
# ---------------------------------------------------------------------------


def classify_valuation(price: float | None, low: float, high: float) -> str:
    """Map spot price vs [low, high] fair band to a short English label."""
    if price is None or price <= 0:
        return "Unable to classify (no price)"
    if price < low:
        return "Materially undervalued (below fair band)"
    if price <= high:
        return "Fair (within valuation band)"
    return "Overvalued (above fair band)"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Stock Valuation Dashboard", layout="wide")
    st.title("Stock Valuation Dashboard")
    st.caption(
        "Data: yfinance — simplified two-stage DCF for education/research only, not investment advice."
    )

    with st.sidebar:
        st.header("Ticker & DCF inputs")
        ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper() or "NVDA"
        g5 = st.slider("5-year FCF growth rate (annual, %)", 0, 100, 20, format="%d%%") / 100.0
        g_term = st.slider("Terminal growth rate (%)", 1, 5, 3, format="%d%%") / 100.0
        wacc = st.slider("Discount rate WACC (%)", 5, 20, 10, format="%d%%") / 100.0

    market: MarketData | None = None
    fundamentals: FundamentalInputs | None = None

    try:
        info, t = fetch_ticker_info(ticker)
        hist = t.history(period="5d")
        close_px = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        market = extract_market_data(info, close_px)
        fundamentals = build_fundamental_inputs(t, info)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.info("Check the ticker, network, or try again later.")
        return

    assert market is not None and fundamentals is not None

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
        dcf_error = f"DCF error: {e}"

    st.subheader("1. Fundamentals & relative valuation")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Last price (USD)", f"${market.price:,.2f}" if market.price else "—")
    with c2:
        st.metric("Trailing P/E", f"{market.trailing_pe:.2f}" if market.trailing_pe else "—")
    with c3:
        st.metric("Forward P/E", f"{market.forward_pe:.2f}" if market.forward_pe else "—")
    with c4:
        st.metric("PEG", f"{market.peg:.2f}" if market.peg else "—")
    with c5:
        st.metric("P/B", f"{market.pb:.2f}" if market.pb else "—")

    if fundamentals.used_defaults:
        with st.expander("Data gaps (defaults or substitutes applied)", expanded=False):
            for line in fundamentals.used_defaults:
                st.warning(line)

    st.divider()

    st.subheader("2. Absolute valuation — two-stage DCF")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
| Item | Value |
|------|-------|
| Base-year FCF (implied) | {fundamentals.fcf_base:,.0f} USD |
| Shares outstanding | {fundamentals.shares:,.0f} |
| Net debt (debt − cash) | {fundamentals.net_debt:,.0f} USD |
"""
        )
    with col_b:
        st.markdown(
            f"""
| DCF input | Setting |
|-----------|---------|
| 5-year FCF growth | {g5*100:.1f}% |
| Terminal growth | {g_term*100:.1f}% |
| WACC | {wacc*100:.1f}% |
"""
        )

    if dcf_error:
        st.error(dcf_error)
    elif dcf is not None:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Intrinsic value / share (est.)", f"${dcf.intrinsic_per_share:,.2f}")
        with m2:
            st.metric("Fair band low (−10%)", f"${dcf.band_low:,.2f}")
        with m3:
            st.metric("Fair band high (+10%)", f"${dcf.band_high:,.2f}")
        st.caption(
            f"EV ≈ {dcf.enterprise_value:,.0f} USD — PV(5y FCF) {dcf.pv_fcf_5y:,.0f} — PV(terminal) {dcf.pv_terminal:,.0f}"
        )
    else:
        st.warning("DCF could not be completed — check inputs and data.")

    st.divider()

    st.subheader("3. Margin of safety vs spot price")
    if dcf is not None and market.price:
        label = classify_valuation(market.price, dcf.band_low, dcf.band_high)
        st.markdown(f"### Verdict: **{label}**")

        chart_df = pd.DataFrame(
            {
                "Item": ["Fair band low", "Spot price", "Intrinsic / share", "Fair band high"],
                "USD": [dcf.band_low, market.price, dcf.intrinsic_per_share, dcf.band_high],
            }
        ).set_index("Item")
        st.bar_chart(chart_df)

        lo = min(dcf.band_low * 0.85, market.price * 0.5)
        hi = max(dcf.band_high * 1.15, market.price * 1.5)
        if hi > lo:
            pct = (market.price - lo) / (hi - lo)
            pct = max(0.0, min(1.0, pct))
        else:
            pct = 0.5
        st.progress(pct)
        st.caption(f"Progress bar: relative spot price on a schematic scale (~{lo:.1f}–{hi:.1f} USD).")
    else:
        st.info("Need both spot price and DCF output to show margin-of-safety chart.")


if __name__ == "__main__":
    main()
