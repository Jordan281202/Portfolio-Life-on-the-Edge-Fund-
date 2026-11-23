"""Risk and performance attribution for the Life on the Edge Fund dashboard.

Answers one question honestly: how much of the result comes from simply
holding more market risk (beta) than QQQ, and how much from stock selection?

All functions are pure: they take pandas objects and plain parameters and
return pandas objects / dicts, so they can be tested in isolation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def capm_decomposition(ann_return: float, ann_bench: float, beta: float, rf: float) -> dict:
    """Split the annualised return into risk-free, beta-driven and alpha parts (CAPM)."""
    market_part = beta * (ann_bench - rf)
    alpha = ann_return - rf - market_part
    return {
        "rf": rf,
        "market": market_part,
        "alpha": alpha,
        "total": ann_return,
        # What QQQ "would have delivered" with the same beta, before any selection.
        "beta_matched_return": rf + market_part,
    }


def beta_matched_benchmark(bret: pd.Series, beta: float, rf: float, af: int = 252) -> pd.Series:
    """Daily returns of QQQ re-levered to the portfolio's beta (rf + beta * (QQQ - rf)).

    If the portfolio only beat QQQ because of its higher beta, this line would
    sit on top of the portfolio. The gap between the two is stock selection.
    """
    rfd = rf / af
    out = rfd + beta * (bret - rfd)
    out.iloc[0] = 0.0
    return out


def rolling_alpha(ret: pd.Series, bret: pd.Series, window: int, rf: float, af: int = 252) -> pd.DataFrame:
    """Rolling annualised Jensen's alpha and beta over `window` trading days."""
    rfd = rf / af
    beta = ret.rolling(window).cov(bret) / bret.rolling(window).var()
    alpha_d = (ret.rolling(window).mean() - rfd) - beta * (bret.rolling(window).mean() - rfd)
    return pd.DataFrame({"alpha": alpha_d * af, "beta": beta}).dropna()


def risk_contribution(prices: pd.DataFrame, weights: pd.Series, af: int = 252) -> pd.DataFrame:
    """Ex-post contribution of each position to portfolio volatility.

    Uses current weights and the full-period daily covariance matrix:
    RC_i = w_i * (Sigma w)_i / (w' Sigma w). The RC_i sum to 100%.
    """
    tickers = [t for t in weights.index if t in prices.columns]
    rets = prices[tickers].pct_change().dropna(how="all").fillna(0.0)
    w = weights.reindex(tickers).fillna(0.0).values
    cov = rets.cov().values * af
    port_var = float(w @ cov @ w)
    marginal = cov @ w
    rc = w * marginal / port_var if port_var > 0 else np.zeros_like(w)
    vols = np.sqrt(np.diag(cov))
    port_vol = math.sqrt(port_var) if port_var > 0 else float("nan")
    df = pd.DataFrame({
        "ticker": tickers,
        "weight": w,
        "vol": vols,
        "risk_contrib": rc,
    })
    df.attrs["port_vol"] = port_vol
    df.attrs["div_ratio"] = float((w * vols).sum() / port_vol) if port_vol and port_vol > 0 else float("nan")
    return df


def selection_vs_beta(pos: pd.DataFrame, bench_total_return: float, rf: float, sessions: int, af: int = 252) -> pd.DataFrame:
    """Per position: P&L explained by beta vs P&L from stock selection (single-period CAPM).

    beta-implied return_i = rf_period + beta_i * (QQQ return - rf_period)
    selection P&L_i       = actual P&L_i - cost_i * beta-implied return_i
    """
    rf_period = (1 + rf) ** (sessions / af) - 1
    df = pos[["ticker", "risk_bucket", "beta", "pnl", "market_value"]].copy()
    df["cost"] = pos["quantity"] * pos["buy_price"]
    df["implied_ret"] = rf_period + df["beta"].fillna(1.0) * (bench_total_return - rf_period)
    df["beta_pnl"] = df["cost"] * df["implied_ret"]
    df["selection_pnl"] = df["pnl"] - df["beta_pnl"]
    return df.sort_values("selection_pnl", ascending=False).reset_index(drop=True)


def bucket_summary(pos: pd.DataFrame, rc: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    """Risk budget vs result per risk bucket (CORE / GROWTH / SPECULATIVE)."""
    m = pos[["ticker", "risk_bucket", "weight", "pnl"]].merge(rc[["ticker", "risk_contrib"]], on="ticker", how="left")
    m = m.merge(sel[["ticker", "selection_pnl"]], on="ticker", how="left")
    out = m.groupby("risk_bucket", as_index=False).agg(
        n=("ticker", "count"),
        weight=("weight", "sum"),
        risk_contrib=("risk_contrib", "sum"),
        pnl=("pnl", "sum"),
        selection_pnl=("selection_pnl", "sum"),
    )
    order = {"CORE": 0, "GROWTH": 1, "SPECULATIVE": 2}
    return out.sort_values("risk_bucket", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


def window_stats(nav: pd.Series, bench_nav: pd.Series, start: str, end: str | None) -> dict | None:
    """Portfolio vs QQQ over a dated window, used to annotate journal entries with real data."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end else nav.index[-1]
    nv = nav.loc[(nav.index >= s) & (nav.index <= e)]
    bv = bench_nav.loc[(bench_nav.index >= s) & (bench_nav.index <= e)]
    if len(nv) < 2 or len(bv) < 2:
        return None
    r = nv.pct_change().dropna()
    b = bv.pct_change().dropna()
    beta = float(r.cov(b) / b.var()) if len(r) > 5 and b.var() else float("nan")
    return {
        "start": nv.index[0],
        "end": nv.index[-1],
        "port_ret": float(nv.iloc[-1] / nv.iloc[0] - 1),
        "bench_ret": float(bv.iloc[-1] / bv.iloc[0] - 1),
        "max_dd": float((nv / nv.cummax() - 1).min()),
        "beta": beta,
        "sessions": int(len(nv)),
    }


def calendar_alpha(nav: pd.Series, bench_nav: pd.Series, init: float, rf: float, af: int = 252) -> pd.DataFrame:
    """CAPM attribution year by year: did the fund beat QQQ *at its own beta* in each calendar year?

    period alpha = fund return - [rf_period + beta_year * (QQQ return - rf_period)]
    The first year starts from the capital invested at entry prices.
    """
    rows = []
    for year in sorted(set(nav.index.year)):
        nv = nav[nav.index.year == year]
        bv = bench_nav[bench_nav.index.year == year]
        if len(nv) < 10:
            continue
        prev_n = nav[nav.index < nv.index[0]]
        prev_b = bench_nav[bench_nav.index < bv.index[0]]
        start_n = float(prev_n.iloc[-1]) if len(prev_n) else float(init)
        start_b = float(prev_b.iloc[-1]) if len(prev_b) else float(bv.iloc[0])
        fund = float(nv.iloc[-1] / start_n - 1)
        bench = float(bv.iloc[-1] / start_b - 1)
        r = pd.concat([pd.Series([start_n]), nv.reset_index(drop=True)]).pct_change().dropna()
        b = pd.concat([pd.Series([start_b]), bv.reset_index(drop=True)]).pct_change().dropna()
        beta = float(r.cov(b) / b.var()) if b.var() else float("nan")
        rf_p = (1 + rf) ** (len(nv) / af) - 1
        implied = rf_p + beta * (bench - rf_p)
        rows.append({
            "period": f"{year}" + (" (from inception)" if not len(prev_n) else (" YTD" if year == nav.index[-1].year else "")),
            "fund": fund, "bench": bench, "beta": beta, "implied": implied, "alpha": fund - implied,
        })
    return pd.DataFrame(rows)
