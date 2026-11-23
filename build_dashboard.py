#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

import attribution as attr
from content import load_content

warnings.filterwarnings("ignore")


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Index):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder, ensure_ascii=False)


def _dt_list(idx) -> list[str]:
    return [pd.to_datetime(x).strftime("%Y-%m-%d") for x in idx]


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

OUT_HTML = DOCS / "index.html"
OUT_JSON = DATA / "dashboard_snapshot.json"

CFG = {
    "portfolio_name": "Life on the Edge Fund",
    "manager": "Jordan",
    "github_url": "https://github.com/Jordan281202/Portfolio-Life-on-the-Edge-Fund-",
    "linkedin_url": "",  # add your LinkedIn profile URL here to show it in the header/footer
    "contact_email": "",
    "school": "Trinity College Dublin",
    "school_tag": "#1 in Ireland",
    "course": "Investment Analysis · Academic Portfolio",
    "benchmark": "QQQ",
    "bench2": "SPY",
    "inception": "2025-03-06",
    "journal_written": "2026-11-23",  # date the retrospective journal entries were written
    "rf": 0.0450,
    "af": 252,
    "roll_w": 30,
    "alpha_w": 126,  # ~6 months window for rolling alpha
    "mc_seed": 42,
    "mc_paths": 600,
    "news_n": 10,
}

C = {
    "bg": "#07090F",
    "panel": "#0C1018",
    "card": "#111520",
    "card2": "#161C2A",
    "border": "#1C2840",
    "border2": "#243350",
    "green": "#00D97E",
    "blue": "#2D7DD2",
    "red": "#E8304A",
    "amber": "#F0A500",
    "purple": "#7C5CBF",
    "cyan": "#00B4D8",
    "text": "#D8E2F0",
    "muted": "#7A8FAD",
    "dim": "#3A4A65",
}

SC = {
    "AI / Semiconductors": "#2D7DD2",
    "AI / Tech Platform": "#4D8FE0",
    "AI / Defence Tech": "#7C5CBF",
    "AI / AdTech": "#9169CC",
    "AI / Voice": "#A27AD8",
    "Defense / Aerospace": "#5A76C6",
    "Space Economy": "#7F5BC9",
    "Energy Transition": "#F0A500",
    "Crypto Infrastructure": "#D96B3B",
    "Bitcoin Mining": "#C24D34",
    "Fintech / Retail": "#00D97E",
    "Mobility / Platform": "#00B4D8",
    "Social / AI Data": "#4BC6B6",
}

BUCKET_C = {
    "CORE": C["blue"],
    "GROWTH": C["green"],
    "SPECULATIVE": C["amber"],
}

SCENARIOS = [
    {"name": "Broad market -10%", "type": "benchmark", "shock": -0.10, "desc": "10% QQQ drawdown mapped through beta."},
    {"name": "Growth de-rating", "type": "bucket", "bucket": "GROWTH", "shock": -0.18, "desc": "Growth multiple compression scenario."},
    {"name": "Speculative risk-off", "type": "bucket", "bucket": "SPECULATIVE", "shock": -0.25, "desc": "High-beta / speculative names sell off sharply."},
    {"name": "AI multiple compression", "type": "theme_kw", "kws": ["AI"], "shock": -0.17, "desc": "AI narrative repricing across related positions."},
    {"name": "Rates shock", "type": "custom", "mapping": {"GROWTH": -0.12, "SPECULATIVE": -0.16, "CORE": -0.06}, "desc": "Higher real yields pressure long-duration exposures."},
    {"name": "Crypto crash", "type": "tickers", "tickers": ["COIN", "MARA", "HOOD"], "shock": -0.28, "desc": "Crypto sleeve reprices sharply."},
]


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fc(x: float, d: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{d}f}"


def _fp(x: float, d: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{d}f}%"


def _fpu(x: float, d: int = 2) -> str:
    """Unsigned percentage, for quantities that have no direction (vol, hit ratio, weights)."""
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}%"


def _fx(x: float, d: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}x"


def _col(x: float) -> str:
    return C["green"] if x >= 0 else C["red"]


def _colors(arr, pos_col: str = None, neg_col: str = None) -> list:
    pc = pos_col or C["green"]
    nc = neg_col or C["red"]
    return [pc if float(v) >= 0 else nc for v in arr]


def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"ticker", "name", "quantity", "buy_price", "sector", "theme", "risk_bucket", "inception_date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="raise")
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_buf,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        lv0 = [str(v) for v in raw.columns.get_level_values(0)]
        lv1 = [str(v) for v in raw.columns.get_level_values(1)]

        if "Close" in lv0:
            for t in tickers:
                col = ("Close", t)
                if col in raw.columns:
                    frames.append(raw[col].rename(t))
        elif "Close" in lv1:
            for t in tickers:
                col = (t, "Close")
                if col in raw.columns:
                    frames.append(raw[col].rename(t))

        if not frames:
            raise RuntimeError("Unable to extract Close prices from yfinance response.")

        close = pd.concat(frames, axis=1)
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("No Close column returned by yfinance.")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    close = close.sort_index().ffill(limit=5).dropna(how="all")

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Missing tickers in downloaded data: {missing}")

    return close


def build_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    px = prices.copy()
    bench = CFG["benchmark"]
    bench2 = CFG["bench2"]
    init = float(holdings["cost_basis"].sum())

    pos_mv = pd.DataFrame(index=px.index)
    for _, r in holdings.iterrows():
        t = r["ticker"]
        if t in px.columns:
            pos_mv[t] = px[t] * r["quantity"]

    nav = pos_mv.sum(axis=1)

    qqq_units = init / float(px[bench].dropna().iloc[0])
    qqq_nav = px[bench] * qqq_units

    if bench2 in px.columns and not px[bench2].dropna().empty:
        spy_units = init / float(px[bench2].dropna().iloc[0])
        spy_nav = px[bench2] * spy_units
        s2ret = spy_nav.pct_change().fillna(0)
        s2cum = (1 + s2ret).cumprod()
    else:
        spy_nav = pd.Series(dtype=float, index=px.index)
        s2ret = pd.Series(dtype=float, index=px.index)
        s2cum = pd.Series(dtype=float, index=px.index)

    # Day-1 return is measured from the actual entry prices (cost basis), not from the
    # first close, so that headline P&L, total return and position P&L all reconcile.
    ret = nav.pct_change()
    ret.iloc[0] = float(nav.iloc[0] / init - 1)
    bret = qqq_nav.pct_change().fillna(0)
    cum = nav / init
    bcum = (1 + bret).cumprod()

    peak = np.maximum(nav.cummax(), init)
    dd = (nav / peak - 1) * 100
    bdd = (qqq_nav / qqq_nav.cummax() - 1) * 100

    return {
        "prices": px,
        "init": init,
        "pos_mv": pos_mv,
        "nav": nav,
        "bench_nav": qqq_nav,
        "bench2_nav": spy_nav,
        "ret": ret,
        "bret": bret,
        "b2ret": s2ret,
        "b100": cum * 100,
        "bb100": bcum * 100,
        "b2b100": s2cum * 100,
        "dd": dd,
        "bdd": bdd,
    }


def _ann(total: float, n: int, af: int = 252) -> float:
    if n <= 0:
        return float("nan")
    return (1 + total) ** (af / n) - 1


def _downside(r: pd.Series, mar: float, af: int) -> float:
    d = np.minimum(r - mar, 0)
    return float(np.sqrt(np.mean(d**2)) * np.sqrt(af))


def _omega(r: pd.Series, mar: float) -> float:
    d = r - mar
    g = d[d > 0].sum()
    l = -d[d < 0].sum()
    return float(g / l) if l > 0 else float("nan")


def _capture(port: pd.Series, bench: pd.Series, up: bool) -> float:
    mask = bench > 0 if up else bench < 0
    if mask.sum() == 0:
        return float("nan")
    b = bench[mask].mean()
    if abs(b) < 1e-12:
        return float("nan")
    return float(port[mask].mean() / b)


def compute_metrics(frame: dict) -> dict:
    af = CFG["af"]
    rf = CFG["rf"]
    rfd = rf / af

    r = frame["ret"]
    b = frame["bret"]
    n = len(r)

    init = frame["init"]
    tr = float(frame["nav"].iloc[-1] / init - 1)
    trb = float(frame["bench_nav"].iloc[-1] / frame["bench_nav"].iloc[0] - 1)
    annr = _ann(tr, n, af)
    annb = _ann(trb, n, af)

    vol = float(r.std() * math.sqrt(af))
    bvol = float(b.std() * math.sqrt(af))
    ddev = _downside(r, rfd, af)
    bddev = _downside(b, rfd, af)

    sharpe = (annr - rf) / vol if vol else float("nan")
    sortino = (annr - rf) / ddev if ddev else float("nan")
    beta = float(r.cov(b) / b.var()) if b.var() else float("nan")
    corr = float(r.corr(b))
    jalpha = float(annr - (rf + beta * (annb - rf))) if not math.isnan(beta) else float("nan")

    act = r - b
    te = float(act.std() * math.sqrt(af))
    ir = float((act.mean() * af) / te) if te else float("nan")

    wealth = (1 + r).cumprod()
    dd_s = (wealth / np.maximum(wealth.cummax(), 1.0) - 1)
    mdd = float(dd_s.min())
    bwealth = (1 + b).cumprod()
    bmdd = float((bwealth / bwealth.cummax() - 1).min())
    calmar = float(annr / abs(mdd)) if mdd != 0 else float("nan")

    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean())
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    treynor = float((annr - rf) / beta) if (not math.isnan(beta) and abs(beta) > 1e-12) else float("nan")
    omega = _omega(r, rfd)
    upc = _capture(r, b, True)
    dnc = _capture(r, b, False)
    hit = float((r > 0).mean())
    bsharpe = (annb - rf) / bvol if bvol else float("nan")
    # Modigliani M²: portfolio return if it were scaled to QQQ's volatility.
    m2 = rf + sharpe * bvol if not math.isnan(sharpe) else float("nan")

    dpnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[-2])
    dret = float(r.iloc[-1])
    tpnl = float(frame["nav"].iloc[-1] - init)

    roll = CFG["roll_w"]
    rvol = r.rolling(roll).std() * math.sqrt(af) * 100
    rbeta = r.rolling(roll).cov(b) / b.rolling(roll).var()
    rsh = ((r.rolling(roll).mean() - rfd) / r.rolling(roll).std()) * math.sqrt(af)

    mp = (1 + r).resample("ME").prod() - 1
    mb = (1 + b).resample("ME").prod() - 1

    return dict(
        current_nav=float(frame["nav"].iloc[-1]),
        daily_pnl=dpnl,
        daily_return=dret,
        total_pnl=tpnl,
        total_return=tr,
        bench_total_return=trb,
        alpha=tr - trb,
        ann_return=annr,
        ann_bench=annb,
        vol=vol,
        bvol=bvol,
        ddev=ddev,
        bddev=bddev,
        sharpe=sharpe,
        sortino=sortino,
        beta=beta,
        corr=corr,
        jalpha=jalpha,
        te=te,
        ir=ir,
        mdd=mdd,
        bmdd=bmdd,
        calmar=calmar,
        var95=var95,
        cvar95=cvar95,
        skew=skew,
        kurt=kurt,
        treynor=treynor,
        omega=omega,
        upc=upc,
        dnc=dnc,
        hit=hit,
        bsharpe=bsharpe,
        m2=m2,
        sessions=n,
        rvol=rvol,
        rbeta=rbeta,
        rsh=rsh,
        mp=mp,
        mb=mb,
    )


def compute_positions(frame: dict, holdings: pd.DataFrame) -> pd.DataFrame:
    px = frame["prices"]
    bret = frame["bret"]
    nav = float(frame["nav"].iloc[-1])
    init_aum = float(holdings["cost_basis"].sum())
    rows = []

    for _, h in holdings.iterrows():
        t = h["ticker"]
        if t not in px.columns:
            continue
        s = px[t].dropna()
        r = s.pct_change().dropna()
        p = float(s.iloc[-1])
        pp = float(s.iloc[-2]) if len(s) >= 2 else p
        mv = p * float(h["quantity"])
        cb = float(h["cost_basis"])
        pnl = mv - cb

        def trail(d: int) -> float:
            return float(s.iloc[-1] / s.iloc[-d - 1] - 1) if len(s) > d else float("nan")

        aligned = pd.concat([r, bret], axis=1, join="inner").dropna()
        aligned.columns = ["r", "b"]
        beta_i = float(aligned["r"].cov(aligned["b"]) / aligned["b"].var()) if len(aligned) > 10 and aligned["b"].var() else float("nan")

        rows.append(dict(
            ticker=t,
            name=h["name"],
            sector=h["sector"],
            theme=h["theme"],
            risk_bucket=h["risk_bucket"],
            quantity=float(h["quantity"]),
            buy_price=float(h["buy_price"]),
            latest_price=p,
            market_value=mv,
            pnl=pnl,
            ret=pnl / cb,
            weight=mv / nav,
            contribution=pnl / init_aum,
            d1=p / pp - 1,
            d5=trail(5),
            d21=trail(21),
            beta=beta_i,
        ))

    return pd.DataFrame(rows).sort_values("market_value", ascending=False).reset_index(drop=True)


def compute_structure(pos: pd.DataFrame) -> dict:
    sector = pos.groupby("sector", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    theme = pos.groupby("theme", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    bucket = pos.groupby("risk_bucket", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    hhi = float((pos["weight"] ** 2).sum())
    eff_n = float(1 / hhi) if hhi else float("nan")
    top5 = float(pos["weight"].head(5).sum())

    return dict(sector=sector, theme=theme, bucket=bucket, hhi=hhi, eff_n=eff_n, top5=top5)


def build_ledger(frame: dict) -> pd.DataFrame:
    nav = frame["nav"]
    df = pd.DataFrame({
        "date": nav.index,
        "nav": nav.values,
        "daily_pnl": nav.diff().fillna(0).values,
        "daily_ret": frame["ret"].values,
        "bench_ret": frame["bret"].values,
        "active": (frame["ret"] - frame["bret"]).values,
        "dd": frame["dd"].values / 100,
    })
    return df.tail(40).iloc[::-1].reset_index(drop=True)


def build_heatmap(mp: pd.Series) -> pd.DataFrame:
    df = mp.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")
    pivot = df.pivot(index="year", columns="month", values="r")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=[m for m in order if m in pivot.columns])


def build_stress(pos: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    nav = pos["market_value"].sum()
    beta = metrics["beta"]
    rows = []
    for s in SCENARIOS:
        shock = pos[["ticker", "market_value", "risk_bucket", "theme"]].copy()
        shock["sh"] = 0.0

        if s["type"] == "benchmark":
            shock["sh"] = beta * s["shock"]
        elif s["type"] == "bucket":
            shock.loc[shock["risk_bucket"] == s["bucket"], "sh"] = s["shock"]
        elif s["type"] == "theme_kw":
            mask = shock["theme"].str.contains("|".join(s["kws"]), case=False, na=False)
            shock.loc[mask, "sh"] = s["shock"]
        elif s["type"] == "tickers":
            shock.loc[shock["ticker"].isin(s["tickers"]), "sh"] = s["shock"]
        elif s["type"] == "custom":
            shock["sh"] = shock["risk_bucket"].map(s["mapping"]).fillna(0.0)

        total = (shock["market_value"] * shock["sh"]).sum()
        rows.append(dict(
            scenario=s["name"],
            desc=s["desc"],
            pnl_impact=total,
            ret_impact=total / nav,
            nav_after=nav + total,
        ))
    return pd.DataFrame(rows).sort_values("pnl_impact")


def build_forecast(frame: dict):
    np.random.seed(CFG["mc_seed"])
    r = frame["ret"].dropna()
    nav0 = float(frame["nav"].iloc[-1])

    mu = float(r.mean())
    sig = float(r.std())

    horizons = {"3M": 63, "6M": 126, "12M": 252}
    sum_rows, path_rows = [], []

    for lbl, h in horizons.items():
        shocks = np.random.normal(mu, sig, (CFG["mc_paths"], h))
        wealth = nav0 * np.cumprod(1 + shocks, axis=1)
        ending = wealth[:, -1]

        sum_rows.append(dict(
            horizon=lbl,
            start_nav=nav0,
            p05=float(np.percentile(ending, 5)),
            p25=float(np.percentile(ending, 25)),
            median=float(np.percentile(ending, 50)),
            p75=float(np.percentile(ending, 75)),
            p95=float(np.percentile(ending, 95)),
        ))

        steps = list(range(h + 1))
        nav_col = np.full((CFG["mc_paths"], 1), nav0)
        all_w = np.hstack([nav_col, wealth])

        mc_low = np.percentile(all_w, 5, axis=0).tolist()
        mc_high = np.percentile(all_w, 95, axis=0).tolist()

        bull = (nav0 * np.cumprod(np.r_[1, np.repeat(mu + 0.5 * sig, h)])).tolist()
        base = (nav0 * np.cumprod(np.r_[1, np.repeat(mu, h)])).tolist()
        bear = (nav0 * np.cumprod(np.r_[1, np.repeat(mu - 0.5 * sig, h)])).tolist()

        for i, step in enumerate(steps):
            path_rows.append(dict(
                horizon=lbl,
                step=step,
                bull=bull[i],
                base=base[i],
                bear=bear[i],
                mc_low=mc_low[i],
                mc_high=mc_high[i],
            ))

    return pd.DataFrame(sum_rows), pd.DataFrame(path_rows)


NEWS_ALIASES = {
    "GOOGL": ["alphabet", "google", "gemini", "waymo"],
    "NVDA": ["nvidia", "jensen huang"],
    "PLTR": ["palantir"],
    "APP": ["applovin"],
    "SOUN": ["soundhound"],
    "RTX": ["rtx", "raytheon", "pratt"],
    "RKLB": ["rocket lab", "neutron"],
    "GEV": ["ge vernova", "vernova"],
    "COIN": ["coinbase"],
    "MARA": ["mara holdings", "marathon digital", "mara "],
    "HOOD": ["robinhood"],
    "UBER": ["uber"],
    "RDDT": ["reddit"],
}


def _news_is_relevant(ticker: str, title: str) -> bool:
    """Yahoo's feed often attaches unrelated stories to a ticker; keep only headlines that name the company."""
    t = f" {str(title).lower()} "
    keys = NEWS_ALIASES.get(ticker, []) + [ticker.lower()]
    return any(re.search(rf"\b{re.escape(k.strip())}\b", t) for k in keys)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    items = []
    for ticker in holdings["ticker"].tolist():
        try:
            tk = yf.Ticker(ticker)
            news = getattr(tk, "news", None) or []
            for a in news[:5]:
                ct = a.get("content", {}) if isinstance(a, dict) else {}
                ttl = ct.get("title") or a.get("title")
                url = ct.get("canonicalUrl", {}).get("url") or a.get("link") or a.get("url")
                src = ct.get("provider", {}).get("displayName") or a.get("publisher") or "Yahoo Finance"
                pub = ct.get("pubDate") or a.get("providerPublishTime")

                if isinstance(pub, str):
                    pub = pd.to_datetime(pub, utc=True, errors="coerce")
                elif pub is not None:
                    pub = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
                else:
                    pub = pd.NaT

                if ttl and url and _news_is_relevant(ticker, ttl):
                    items.append(dict(
                        ticker=ticker,
                        title=ttl,
                        source=src,
                        published_at=pub,
                        url=url,
                    ))
        except Exception:
            continue

    if not items:
        return pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])

    df = pd.DataFrame(items)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    df["_norm"] = df["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    return df.head(CFG["news_n"]).reset_index(drop=True)


def build_intelligence(metrics: dict, pos: pd.DataFrame, structure: dict, stress: pd.DataFrame) -> list:
    top3 = pos.nlargest(3, "contribution")
    bot3 = pos.nsmallest(3, "contribution")
    dsec = structure["sector"].iloc[0]
    wstress = stress.iloc[0]

    return [
        ("Portfolio DNA",
         f"Concentrated US equity book with high-conviction thematic growth bias. Top 5 positions represent {structure['top5']*100:.1f}% of NAV. HHI {structure['hhi']:.3f} → effective position count {structure['eff_n']:.1f}."),
        ("What worked",
         f"Top contribution came from {', '.join(top3['ticker'].tolist())}. Main detractors were {', '.join(bot3['ticker'].tolist())}."),
        ("Risk lens",
         f"Beta {metrics['beta']:.2f}x vs {CFG['benchmark']}, annualized vol {metrics['vol']*100:.1f}%, max drawdown {metrics['mdd']*100:.1f}%. Worst modelled shock: '{wstress['scenario']}' ({wstress['ret_impact']*100:.1f}% NAV hit)."),
        ("Benchmark lens",
         f"Excess return vs {CFG['benchmark']}: {metrics['alpha']*100:+.1f}% since inception, while Jensen's alpha is {metrics['jalpha']*100:+.1f}% a year: "
         f"{'stock selection added value beyond the extra beta' if metrics['jalpha'] > 0 else 'most of the outperformance is explained by the higher beta, not by selection'}. "
         f"Risk-adjusted (M²) the book returns {metrics['m2']*100:.1f}% vs {metrics['ann_bench']*100:.1f}% for {CFG['benchmark']}. "
         f"Upside capture {metrics['upc']:.2f}x, downside capture {metrics['dnc']:.2f}x."),
        ("Concentration lens",
         f"Dominant sector: {dsec['sector']} at {dsec['weight']*100:.1f}% NAV. No rebalancing means winners are allowed to drift higher in weight over time."),
    ]


def _layout(title: str, h: int = 360) -> dict:
    return dict(
        # Card headers already carry the chart titles; an in-chart title would collide with the legend.
        title=dict(text=""),
        meta=dict(title=title),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["card"],
        margin=dict(l=56, r=18, t=40, b=42),
        font=dict(family="DM Sans, Arial, sans-serif", size=11, color=C["text"]),
        xaxis=dict(
            type="date",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="outside",
            ticklen=4,
            tickcolor=C["border"],
            tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
            tickformat="%b %Y",
            hoverformat="%d %b %Y",
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="outside",
            ticklen=4,
            tickcolor=C["border"],
            tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
        ),
        hoverlabel=dict(
            bgcolor=C["card2"],
            bordercolor=C["border2"],
            font=dict(color=C["text"], size=11, family="DM Sans, Arial, sans-serif"),
        ),
        hovermode="x unified",
        height=h,
    )


def make_charts(frame: dict, metrics: dict, pos: pd.DataFrame, structure: dict, heatmap: pd.DataFrame, stress: pd.DataFrame, fp: pd.DataFrame) -> dict:
    charts = {}

    portfolio_color = "#00D97E"
    benchmark_color = "#2D7DD2"
    benchmark2_color = "#7C5CBF"
    down_fill = "rgba(232,48,74,0.14)"
    green_soft = "rgba(0,217,126,0.14)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(frame["b100"].index), y=frame["b100"].tolist(), name=CFG["portfolio_name"], line=dict(color=portfolio_color, width=3.4)))
    fig.add_trace(go.Scatter(x=_dt_list(frame["bb100"].index), y=frame["bb100"].tolist(), name=CFG["benchmark"], line=dict(color=benchmark_color, width=2.1)))
    if not frame["b2b100"].empty:
        fig.add_trace(go.Scatter(x=_dt_list(frame["b2b100"].index), y=frame["b2b100"].tolist(), name=CFG["bench2"], line=dict(color=benchmark2_color, width=1.5, dash="dot")))
    fig.update_layout(**_layout("Portfolio vs Benchmarks — Base 100", 420))
    fig.update_yaxes(title_text="Base 100")
    charts["perf"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(frame["dd"].index), y=frame["dd"].tolist(), name=CFG["portfolio_name"], fill="tozeroy", fillcolor=down_fill, line=dict(color=C["red"], width=2.2)))
    fig.add_trace(go.Scatter(x=_dt_list(frame["bdd"].index), y=frame["bdd"].tolist(), name=CFG["benchmark"], line=dict(color=benchmark_color, width=1.7)))
    fig.update_layout(**_layout("Drawdown from Peak", 330))
    fig.update_yaxes(title_text="%")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.10)", width=1))
    charts["drawdown"] = fig.to_plotly_json()

    mdf = pd.DataFrame({"date": _dt_list(metrics["mp"].index), "val": (metrics["mp"].values * 100)})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mdf["date"].tolist(), y=mdf["val"].tolist(), name=CFG["portfolio_name"], marker_color=_colors(mdf["val"], pos_col=portfolio_color, neg_col=C["red"]), marker_line_width=0, opacity=0.90))
    fig.add_trace(go.Scatter(x=_dt_list(metrics["mb"].index), y=(metrics["mb"].values * 100).tolist(), name=CFG["benchmark"], line=dict(color=benchmark_color, width=2.0)))
    fig.update_layout(**_layout("Monthly Returns vs QQQ", 330))
    fig.update_yaxes(title_text="%")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.10)", width=1))
    charts["monthly_returns"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(metrics["rvol"].dropna().index), y=metrics["rvol"].dropna().tolist(), name="Portfolio vol", line=dict(color=C["amber"], width=2.1)))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Volatility (Ann.)", 320))
    fig.update_yaxes(title_text="%")
    charts["rolling_vol"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(metrics["rbeta"].dropna().index), y=metrics["rbeta"].dropna().tolist(), name="Rolling beta", line=dict(color=C["cyan"], width=2.1)))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.16)", dash="dot", width=1))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Beta vs {CFG['benchmark']}", 320))
    charts["rolling_beta"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(metrics["rsh"].dropna().index), y=metrics["rsh"].dropna().tolist(), name="Rolling Sharpe", line=dict(color=portfolio_color, width=2.1)))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.16)", dash="dot", width=1))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Sharpe", 320))
    charts["rolling_sharpe"] = fig.to_plotly_json()

    fig = go.Figure(data=go.Heatmap(
        z=(heatmap * 100).values.tolist(),
        x=list(heatmap.columns),
        y=[str(y) for y in heatmap.index],
        colorscale=[[0.0, "#6E2030"], [0.48, "#121926"], [0.50, "#161C2A"], [0.52, "#143124"], [1.0, "#00D97E"]],
        zmid=0,
        text=[[("" if pd.isna(v) else f"{v*100:+.1f}%") for v in row] for row in heatmap.values],
        texttemplate="%{text}",
        textfont={"color": "#D8E2F0", "size": 11},
        xgap=1,
        ygap=1,
        colorbar=dict(thickness=10, tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"), outlinewidth=0),
    ))
    fig.update_layout(**_layout("Monthly Return Heatmap", 310))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    charts["heatmap"] = fig.to_plotly_json()

    top_pos = pos.head(10).copy().sort_values("weight", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=(top_pos["weight"] * 100).tolist(),
        y=top_pos["ticker"].tolist(),
        orientation="h",
        marker_color=[BUCKET_C.get(x, C["blue"]) for x in top_pos["risk_bucket"]],
        marker_line_width=0,
        text=[f"{w*100:.1f}%" for w in top_pos["weight"]],
        textposition="outside",
    )])
    fig.update_layout(**_layout("Top Weights", 340))
    fig.update_xaxes(title_text="% NAV", type="linear")
    fig.update_yaxes(type="category")
    charts["top_weights"] = fig.to_plotly_json()

    sector_top = structure["sector"].copy().head(8)
    sector_rest = structure["sector"].copy().iloc[8:]
    if not sector_rest.empty:
        sector_top = pd.concat([sector_top, pd.DataFrame([{
            "sector": "Other",
            "weight": sector_rest["weight"].sum(),
            "market_value": sector_rest["market_value"].sum(),
            "pnl": sector_rest["pnl"].sum(),
            "n": sector_rest["n"].sum(),
        }])], ignore_index=True)

    fig = go.Figure(data=[go.Pie(
        labels=sector_top["sector"].tolist(),
        values=(sector_top["weight"] * 100).tolist(),
        hole=0.62,
        marker=dict(colors=[SC.get(s, C["blue"]) for s in sector_top["sector"]], line=dict(width=1, color=C["bg"])),
        sort=False,
        textinfo="percent",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    )])
    fig.update_layout(**_layout("Sector Allocation", 350))
    charts["sector_alloc"] = fig.to_plotly_json()

    theme_top = structure["theme"].copy().head(8)
    theme_rest = structure["theme"].copy().iloc[8:]
    if not theme_rest.empty:
        theme_top = pd.concat([theme_top, pd.DataFrame([{
            "theme": "Other",
            "weight": theme_rest["weight"].sum(),
            "market_value": theme_rest["market_value"].sum(),
            "pnl": theme_rest["pnl"].sum(),
            "n": theme_rest["n"].sum(),
        }])], ignore_index=True)

    theme_colors = ["#2D7DD2", "#7C5CBF", "#00D97E", "#F0A500", "#00B4D8", "#7A8FAD", "#95A7C2", "#4E79B7", "#8DA0C1"]
    fig = go.Figure(data=[go.Pie(
        labels=theme_top["theme"].tolist(),
        values=(theme_top["weight"] * 100).tolist(),
        hole=0.62,
        marker=dict(colors=theme_colors[:len(theme_top)], line=dict(width=1, color=C["bg"])),
        sort=False,
        textinfo="percent",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    )])
    fig.update_layout(**_layout("Thematic Allocation", 350))
    charts["theme_alloc"] = fig.to_plotly_json()

    pnl_pos = pos.copy().sort_values("pnl", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=pnl_pos["pnl"].tolist(),
        y=pnl_pos["ticker"].tolist(),
        orientation="h",
        marker_color=_colors(pnl_pos["pnl"], pos_col=portfolio_color, neg_col=C["red"]),
        marker_line_width=0,
    )])
    fig.update_layout(**_layout("Position P&L Attribution", 360))
    fig.update_xaxes(title_text="$", type="linear")
    fig.update_yaxes(type="category")
    charts["pnl_attr"] = fig.to_plotly_json()

    stress_plot = stress.copy().sort_values("pnl_impact", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=stress_plot["pnl_impact"].tolist(),
        y=stress_plot["scenario"].tolist(),
        orientation="h",
        marker_color=_colors(stress_plot["pnl_impact"], pos_col=portfolio_color, neg_col=C["red"]),
        marker_line_width=0,
    )])
    fig.update_layout(**_layout("Stress Test P&L Impact", 350))
    fig.update_xaxes(title_text="$", type="linear")
    fig.update_yaxes(type="category")
    charts["stress"] = fig.to_plotly_json()

    fp12 = fp[fp["horizon"] == "12M"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_high"].tolist(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_low"].tolist(), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=green_soft, name="MC 5–95%"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bull"].tolist(), name="Bull", line=dict(color=portfolio_color, width=2.0)))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["base"].tolist(), name="Base", line=dict(color=benchmark_color, width=2.0)))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bear"].tolist(), name="Bear", line=dict(color=C["red"], width=2.0)))
    fig.update_layout(**_layout("12M Scenario Envelope", 340))
    fig.update_xaxes(type="linear", title_text="Trading days")
    fig.update_yaxes(title_text="$")
    charts["forecast"] = fig.to_plotly_json()

    return charts


def build_attribution(frame: dict, metrics: dict, pos: pd.DataFrame) -> dict:
    """Everything needed to answer: 'is the outperformance just beta?'"""
    rf, af = CFG["rf"], CFG["af"]
    capm = attr.capm_decomposition(metrics["ann_return"], metrics["ann_bench"], metrics["beta"], rf)
    bm = attr.beta_matched_benchmark(frame["bret"], metrics["beta"], rf, af)
    bm100 = (1 + bm).cumprod() * 100
    ra = attr.rolling_alpha(frame["ret"], frame["bret"], CFG["alpha_w"], rf, af)
    weights = pos.set_index("ticker")["weight"]
    rc = attr.risk_contribution(frame["prices"], weights, af)
    sel = attr.selection_vs_beta(pos, metrics["bench_total_return"], rf, metrics["sessions"], af)
    buckets = attr.bucket_summary(pos, rc, sel)
    share_alpha_positive = float((ra["alpha"] > 0).mean()) if not ra.empty else float("nan")
    return dict(capm=capm, bm100=bm100, ra=ra, rc=rc, sel=sel, buckets=buckets,
                share_alpha_positive=share_alpha_positive,
                port_vol_rc=rc.attrs.get("port_vol"), div_ratio=rc.attrs.get("div_ratio"))


def journal_with_stats(journal: list, frame: dict) -> list:
    out = []
    for e in journal:
        e = dict(e)
        if e.get("period_start"):
            e["stats"] = attr.window_stats(frame["nav"], frame["bench_nav"], e["period_start"], e.get("period_end"))
        out.append(e)
    return out


def make_attr_charts(frame: dict, metrics: dict, at: dict, journal: list) -> dict:
    charts = {}
    pc, bc, bmc = "#00D97E", "#2D7DD2", "#F0A500"

    # 1. Portfolio vs QQQ vs QQQ levered to the same beta
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_dt_list(frame["b100"].index), y=frame["b100"].tolist(), name=CFG["portfolio_name"], line=dict(color=pc, width=3)))
    fig.add_trace(go.Scatter(x=_dt_list(at["bm100"].index), y=at["bm100"].tolist(), name=f"{CFG['benchmark']} at β={metrics['beta']:.2f}", line=dict(color=bmc, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=_dt_list(frame["bb100"].index), y=frame["bb100"].tolist(), name=CFG["benchmark"], line=dict(color=bc, width=1.6)))
    fig.update_layout(**_layout("Same risk, different result? Portfolio vs beta-matched QQQ", 400))
    fig.update_yaxes(title_text="Base 100")
    charts["beta_matched"] = fig.to_plotly_json()

    # 2. CAPM decomposition of annualised return
    c = at["capm"]
    labels = ["Risk-free", f"Beta × market (β={metrics['beta']:.2f})", "Jensen's alpha (selection)", "Total ann. return"]
    vals = [c["rf"] * 100, c["market"] * 100, c["alpha"] * 100, c["total"] * 100]
    fig = go.Figure(go.Waterfall(
        x=labels, y=vals, measure=["relative", "relative", "relative", "total"],
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
        increasing=dict(marker=dict(color=pc)), decreasing=dict(marker=dict(color=C["red"])),
        totals=dict(marker=dict(color=bc)), connector=dict(line=dict(color=C["border2"], width=1)),
    ))
    lay = _layout("Where the annualised return comes from (CAPM)", 360)
    lay["xaxis"] = dict(type="category", tickfont=dict(size=10, color=C["muted"]))
    lay["hovermode"] = "closest"
    lay["showlegend"] = False
    fig.update_layout(**lay)
    fig.update_yaxes(title_text="% p.a.")
    charts["capm"] = fig.to_plotly_json()

    # 3. Rolling alpha
    ra = at["ra"]
    if not ra.empty:
        a = (ra["alpha"] * 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_dt_list(a.index), y=a.clip(lower=0).tolist(), fill="tozeroy", fillcolor="rgba(0,217,126,0.18)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=_dt_list(a.index), y=a.clip(upper=0).tolist(), fill="tozeroy", fillcolor="rgba(232,48,74,0.18)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=_dt_list(a.index), y=a.tolist(), name=f"Rolling {CFG['alpha_w']}d Jensen's alpha", line=dict(color=C["text"], width=1.8)))
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
        fig.update_layout(**_layout(f"Rolling {CFG['alpha_w']}-day Jensen's alpha (annualised)", 360))
        fig.update_yaxes(title_text="% p.a.")
        charts["rolling_alpha"] = fig.to_plotly_json()

    # 4. Selection vs beta P&L per position
    sel = at["sel"].sort_values("selection_pnl")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sel["ticker"].tolist(), x=sel["beta_pnl"].tolist(), orientation="h", name="Explained by beta", marker=dict(color="rgba(45,125,210,0.55)")))
    fig.add_trace(go.Bar(y=sel["ticker"].tolist(), x=sel["selection_pnl"].tolist(), orientation="h", name="Stock selection", marker=dict(color=_colors(sel["selection_pnl"].tolist()))))
    lay = _layout("P&L per position: beta vs selection", 420)
    lay["xaxis"] = dict(tickfont=dict(size=10, color=C["muted"]), gridcolor="rgba(255,255,255,0.045)", tickprefix="$")
    lay["yaxis"] = dict(type="category", tickfont=dict(size=10, color=C["text"], family="JetBrains Mono, monospace"))
    lay["barmode"] = "relative"
    lay["hovermode"] = "y unified"
    fig.update_layout(**lay)
    charts["selection"] = fig.to_plotly_json()

    # 5. Risk contribution vs weight
    rc = at["rc"].sort_values("risk_contrib", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rc["ticker"].tolist(), y=(rc["weight"] * 100).tolist(), name="Capital weight", marker=dict(color="rgba(45,125,210,0.6)")))
    fig.add_trace(go.Bar(x=rc["ticker"].tolist(), y=(rc["risk_contrib"] * 100).tolist(), name="Share of portfolio risk", marker=dict(color=bmc)))
    lay = _layout("Capital weight vs contribution to volatility", 360)
    lay["xaxis"] = dict(type="category", tickfont=dict(size=10, color=C["text"], family="JetBrains Mono, monospace"))
    lay["barmode"] = "group"
    lay["hovermode"] = "x unified"
    fig.update_layout(**lay)
    fig.update_yaxes(title_text="%", ticksuffix="%")
    charts["risk_contrib"] = fig.to_plotly_json()

    # 6. Journal markers on the main performance chart (added later to charts['perf'])
    marks = []
    for i, e in enumerate(sorted(journal, key=lambda x: x["date"]), start=1):
        d = pd.Timestamp(e["date"])
        if d < frame["b100"].index[0] or d > frame["b100"].index[-1] + pd.Timedelta(days=3):
            continue
        idx = frame["b100"].index.get_indexer([d], method="nearest")[0]
        marks.append(dict(x=frame["b100"].index[idx].strftime("%Y-%m-%d"), y=float(frame["b100"].iloc[idx]), n=i, title=e.get("title", "")))
    charts["_journal_marks"] = marks
    return charts


def _add_journal_marks(perf: dict, marks: list) -> dict:
    if not marks:
        return perf
    perf = dict(perf)
    layout = dict(perf.get("layout", {}))
    anns = list(layout.get("annotations", []) or [])
    for m in marks:
        anns.append(dict(x=m["x"], y=m["y"], text=f"<b>{m['n']}</b>", showarrow=True, arrowhead=0, ax=0, ay=-34,
                         arrowcolor="rgba(240,165,0,0.7)", font=dict(size=10, color="#07090F", family="JetBrains Mono, monospace"),
                         bgcolor="#F0A500", bordercolor="#F0A500", borderpad=3, hovertext=m["title"]))
    layout["annotations"] = anns
    perf["layout"] = layout
    return perf


def _kpi(label: str, value: str, sub: str = "", tone: str = "b") -> str:
    return f"<div class='kpi {tone}'><div class='kpi-label'>{_esc(label)}</div><div class='kpi-val'>{value}</div><div class='kpi-sub'>{_esc(sub)}</div></div>"


def _risk_class(v: str) -> str:
    v = str(v).upper()
    if v == "CORE":
        return "core"
    if v == "GROWTH":
        return "growth"
    return "speculative"


def _positions_table(pos: pd.DataFrame) -> str:
    rows = []
    for _, r in pos.iterrows():
        rows.append(
            f"<tr><td class='td-ticker'>{r['ticker']}</td><td class='td-name'>{_esc(r['name'])}</td><td>{_esc(r['sector'])}</td><td>{_esc(r['theme'])}</td>"
            f"<td><span class='bucket {_risk_class(r['risk_bucket'])}'>{_esc(r['risk_bucket'])}</span></td>"
            f"<td class='num'>{r['quantity']:,.0f}</td><td class='num'>{_fc(r['buy_price'],2)}</td><td class='num'>{_fc(r['latest_price'],2)}</td>"
            f"<td class='num'>{_fc(r['market_value'],0)}</td><td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['ret'])}'>{_fp(r['ret']*100,1)}</td><td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num' style='color:{_col(r['d1'])}'>{_fp(r['d1']*100,1)}</td><td class='num' style='color:{_col(r['d5'])}'>{_fp(r['d5']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r['d21'])}'>{_fp(r['d21']*100,1)}</td><td class='num'>{r['beta']:.2f}</td></tr>"
        )
    return "\n".join(rows)


def _metrics_table(m: dict) -> str:
    rows = [
        ("Current NAV", _fc(m["current_nav"],0), "Marked-to-market NAV."),
        ("Total P&L", _fc(m["total_pnl"],0), "NAV minus capital invested at entry prices."),
        ("Total Return", _fp(m["total_return"]*100), "Measured on capital invested (cost basis)."),
        ("Benchmark Return", _fp(m["bench_total_return"]*100), f"{CFG['benchmark']} since inception."),
        ("Alpha vs Benchmark", _fp(m["alpha"]*100), "Simple excess return vs QQQ."),
        ("Annualized Return", _fp(m["ann_return"]*100), "Compounded annualized return."),
        ("Annualized Volatility", _fpu(m["vol"]*100), f"{CFG['benchmark']}: {_fpu(m['bvol']*100)}."),
        ("Sharpe Ratio", f"{m['sharpe']:.3f}", f"{CFG['benchmark']}: {m['bsharpe']:.3f} · rf={CFG['rf']*100:.2f}%"),
        ("M² (Modigliani)", _fp(m["m2"]*100), f"Return scaled to {CFG['benchmark']} vol; compare with {_fp(m['ann_bench']*100)}."),
        ("Sortino Ratio", f"{m['sortino']:.3f}", "Downside-adjusted."),
        ("Calmar Ratio", f"{m['calmar']:.3f}", "Ann. return / |max drawdown|."),
        ("Max Drawdown", _fp(m["mdd"]*100), f"{CFG['benchmark']}: {_fp(m['bmdd']*100)}."),
        ("Beta", f"{m['beta']:.3f}", f"Vs {CFG['benchmark']}."),
        ("Correlation", f"{m['corr']:.3f}", f"To {CFG['benchmark']}."),
        ("Jensen Alpha", _fp(m["jalpha"]*100), "CAPM alpha: return above what beta alone explains."),
        ("Tracking Error", _fpu(m["te"]*100), "Annualized active risk."),
        ("Information Ratio", f"{m['ir']:.3f}", "Active return / active risk."),
        ("VaR 95% (1d)", _fp(m["var95"]*100), "Historical 5th percentile."),
        ("CVaR 95% (1d)", _fp(m["cvar95"]*100), "Expected shortfall."),
        ("Downside Deviation", _fpu(m["ddev"]*100), "Annualized downside vol."),
        ("Treynor Ratio", f"{m['treynor']:.3f}", "Excess return / beta."),
        ("Omega Ratio", f"{m['omega']:.3f}", "Gain/loss above rf threshold."),
        ("Upside Capture", _fx(m["upc"]), "On benchmark up days."),
        ("Downside Capture", _fx(m["dnc"]), "On benchmark down days."),
        ("Hit Ratio", _fpu(m["hit"]*100), "% positive days."),
        ("Skewness", f"{m['skew']:.3f}", "Return distribution skew."),
        ("Kurtosis", f"{m['kurt']:.3f}", "Tail fatness."),
        ("Sessions", str(m["sessions"]), "Trading days since inception."),
    ]
    return "\n".join(f"<tr><td>{_esc(l)}</td><td class='num'>{v}</td><td>{_esc(n)}</td></tr>" for l, v, n in rows)


def _sector_table(structure: dict) -> str:
    rows = []
    for _, r in structure["sector"].iterrows():
        rows.append(
            f"<tr><td>{_esc(r['sector'])}</td><td class='num'>{r['weight']*100:.1f}%</td><td class='num'>{_fc(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td><td class='num'>{int(r['n'])}</td></tr>"
        )
    return "\n".join(rows)


def _ledger_table(ledger: pd.DataFrame) -> str:
    rows = []
    for _, r in ledger.iterrows():
        rows.append(
            f"<tr><td>{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}</td><td class='num'>{_fc(r['nav'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_pnl'])}'>{_fc(r['daily_pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_ret'])}'>{_fp(r['daily_ret']*100,2)}</td>"
            f"<td class='num'>{_fp(r['bench_ret']*100,2)}</td><td class='num' style='color:{_col(r['active'])}'>{_fp(r['active']*100,2)}</td>"
            f"<td class='num' style='color:{_col(r['dd'])}'>{_fp(r['dd']*100,2)}</td></tr>"
        )
    return "\n".join(rows)


def _stress_table(stress: pd.DataFrame) -> str:
    rows = []
    for _, r in stress.iterrows():
        rows.append(
            f"<tr><td>{_esc(r['scenario'])}</td><td>{_esc(r['desc'])}</td>"
            f"<td class='num' style='color:{_col(r['pnl_impact'])}'>{_fc(r['pnl_impact'],0)}</td>"
            f"<td class='num' style='color:{_col(r['ret_impact'])}'>{_fp(r['ret_impact']*100,1)}</td>"
            f"<td class='num'>{_fc(r['nav_after'],0)}</td></tr>"
        )
    return "\n".join(rows)


def _forecast_table(fs: pd.DataFrame) -> str:
    rows = []
    for _, r in fs.iterrows():
        rows.append(
            f"<tr><td>{_esc(r['horizon'])}</td><td class='num'>{_fc(r['start_nav'],0)}</td><td class='num'>{_fc(r['p05'],0)}</td>"
            f"<td class='num'>{_fc(r['p25'],0)}</td><td class='num'>{_fc(r['median'],0)}</td><td class='num'>{_fc(r['p75'],0)}</td><td class='num'>{_fc(r['p95'],0)}</td></tr>"
        )
    return "\n".join(rows)


def _news_table(news: pd.DataFrame) -> str:
    if news.empty:
        return "<tr><td colspan='4'>News retrieval unavailable. Core analytics unaffected.</td></tr>"
    rows = []
    for _, r in news.iterrows():
        pub = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
        pt = pub.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(pub) else "—"
        rows.append(
            f"<tr><td>{_esc(r['ticker'])}</td><td><a href='{_esc(str(r['url']))}' target='_blank' rel='noopener noreferrer'>{_esc(str(r['title']))}</a></td><td>{_esc(str(r['source']))}</td><td>{pt}</td></tr>"
        )
    return "\n".join(rows)


def _bucket_table(b: pd.DataFrame) -> str:
    rows = []
    for _, r in b.iterrows():
        rows.append(
            f"<tr><td><span class='bucket {_risk_class(r['risk_bucket'])}'>{_esc(r['risk_bucket'])}</span></td><td class='num'>{int(r['n'])}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td><td class='num'>{r['risk_contrib']*100:.1f}%</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['selection_pnl'])}'>{_fc(r['selection_pnl'],0)}</td></tr>"
        )
    return "\n".join(rows)


def _attribution_text(m: dict, at: dict) -> str:
    """Plain-English reading of the attribution numbers, regenerated at every build."""
    c = at["capm"]
    sel = at["sel"]
    pos_sel = sel[sel["selection_pnl"] > 0]
    neg_sel = sel[sel["selection_pnl"] < 0]
    top = ", ".join(pos_sel.head(3)["ticker"]) or "none"
    worst = ", ".join(neg_sel.sort_values("selection_pnl").head(3)["ticker"]) or "none"
    if c["alpha"] > 0:
        verdict = "Stock selection has added value <b>on top of</b> the extra market exposure: the fund did better than a QQQ position levered to the same beta."
    elif m["alpha"] > 0:
        verdict = ("Over the full period, the extra market exposure explains <b>all</b> of the outperformance: a QQQ position levered to the same beta "
                   "would have done slightly better. The winners below show where conviction paid; the detractors show what it cost.")
    else:
        verdict = ("The fund trails both QQQ and a beta-matched QQQ over the full period: the high beta amplified the market's moves "
                   "and selection did not compensate. The detractors below are where the shortfall comes from.")
    share = at["share_alpha_positive"]
    return (
        f"<p>The portfolio runs a beta of <b>{m['beta']:.2f}</b> to {CFG['benchmark']}. With that much market exposure, outperforming in a rising market is expected, "
        f"so the right test is not 'did it beat QQQ?' but 'did it beat QQQ <i>at the same risk</i>?'.</p>"
        f"<p>CAPM splits the {c['total']*100:.1f}% annualised return into {c['rf']*100:.1f}% risk-free, {c['market']*100:+.1f}% from beta × market "
        f"and <b>{c['alpha']*100:+.1f}%</b> of Jensen's alpha. {verdict}</p>"
        f"<p>Selection winners (P&amp;L above what their own beta implied): <b>{_esc(top)}</b>. Main detractors: <b>{_esc(worst)}</b>. "
        f"On a rolling {CFG['alpha_w']}-day basis, alpha was positive {share*100:.0f}% of the time. "
        f"Risk-adjusted, M² is {m['m2']*100:.1f}% vs {m['ann_bench']*100:.1f}% for {CFG['benchmark']}, and the Sharpe ratio is {m['sharpe']:.2f} vs {m['bsharpe']:.2f}.</p>"
    )


def _risk_text(m: dict, at: dict, structure: dict) -> str:
    rc = at["rc"].sort_values("risk_contrib", ascending=False)
    top = rc.head(3)
    top_txt = ", ".join(f"{r.ticker} {r.risk_contrib*100:.0f}% of risk for {r.weight*100:.0f}% of capital" for r in top.itertuples())
    return (
        f"<p>Volatility is not spread evenly: {_esc(top_txt)}. The diversification ratio is <b>{at['div_ratio']:.2f}</b> "
        f"(weighted average stock vol ÷ portfolio vol; 1.0 would mean no diversification at all), and the book behaves like "
        f"<b>{structure['eff_n']:.1f}</b> equally-weighted positions.</p>"
        f"<p>Downside behaviour: max drawdown {m['mdd']*100:.1f}% vs {m['bmdd']*100:.1f}% for {CFG['benchmark']}, which is roughly what a "
        f"{m['beta']:.1f}× beta implies. Upside capture is {m['upc']:.2f}× and downside capture {m['dnc']:.2f}×; "
        f"the gap between the two is what makes high beta worth holding.</p>"
    )


def _thesis_cards(theses: list, pos: pd.DataFrame, sel: pd.DataFrame) -> str:
    if not theses:
        return ""
    p = pos.set_index("ticker")
    s = sel.set_index("ticker")
    cards = []
    intro = ""
    for t in theses:
        if t["key"] == "_intro":
            intro = f"<div class='prose lead'>{t['html']}</div>"
            continue
        k = t["key"]
        chips = ""
        if k in p.index:
            r = p.loc[k]
            sp = s.loc[k, "selection_pnl"] if k in s.index else float("nan")
            chips = (
                f"<div class='chips'><span class='bucket {_risk_class(r['risk_bucket'])}'>{_esc(r['risk_bucket'])}</span>"
                f"<span class='chip'>Weight {r['weight']*100:.1f}%</span>"
                f"<span class='chip' style='color:{_col(r['ret'])}'>Return {_fp(r['ret']*100,1)}</span>"
                f"<span class='chip'>β {r['beta']:.2f}</span>"
                f"<span class='chip' style='color:{_col(sp)}'>Selection {_fc(sp,0)}</span></div>"
            )
        cards.append(f"<details class='thesis'><summary><span class='th-title'>{_esc(t['title'])}</span>{chips}</summary><div class='prose'>{t['html']}</div></details>")
    return intro + "<div class='thesis-grid'>" + "".join(cards) + "</div>"


def _journal_html(journal: list) -> str:
    if not journal:
        return ""
    order = {e["slug"]: i for i, e in enumerate(sorted(journal, key=lambda x: x["date"]), start=1)}
    items = []
    for e in journal:
        st = e.get("stats")
        strip = ""
        if st:
            act = st["port_ret"] - st["bench_ret"]
            strip = (
                f"<div class='chips'><span class='chip'>{st['start'].strftime('%d %b %Y')} → {st['end'].strftime('%d %b %Y')}</span>"
                f"<span class='chip' style='color:{_col(st['port_ret'])}'>Fund {_fp(st['port_ret']*100,1)}</span>"
                f"<span class='chip' style='color:{_col(st['bench_ret'])}'>{CFG['benchmark']} {_fp(st['bench_ret']*100,1)}</span>"
                f"<span class='chip' style='color:{_col(act)}'>Active {_fp(act*100,1)}</span>"
                + (f"<span class='chip'>β {st['beta']:.2f}</span>" if not math.isnan(st['beta']) else "")
                + f"<span class='chip' style='color:var(--red)'>Max DD {_fp(st['max_dd']*100,1)}</span></div>"
            )
        tag = _esc(e.get("tag", ""))
        written = e.get("written", "")
        wnote = f"<span class='written'>written {written}</span>" if written and written != e["date"] else ""
        items.append(
            f"<article class='jentry {tag}' id='j-{_esc(e['slug'])}'><div class='jdot'>{order[e['slug']]}</div>"
            f"<div class='jhead'><span class='jdate'>{pd.Timestamp(e['date']).strftime('%d %b %Y')}</span><span class='jtag {tag}'>{tag}</span>{wnote}</div>"
            f"<h3>{_esc(e.get('title',''))}</h3>{strip}<div class='prose'>{e['html']}</div></article>"
        )
    return "<div class='timeline'>" + "".join(items) + "</div>"


def _sections_html(sections: list, cls: str = "prose") -> str:
    out = []
    for sct in sections:
        head = f"<h3>{_esc(sct['title'])}</h3>" if sct["title"] else ""
        out.append(f"<div class='{cls}'>{head}{sct['html']}</div>")
    return "".join(out)


def generate_html(holdings, metrics, pos, structure, ledger, stress, fcast_sum, news, intel, charts, at=None, content=None) -> str:
    content = content or {"theses": [], "journal": [], "watchlist": [], "multi_asset": [], "about": []}
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    init_aum = float(holdings["cost_basis"].sum())
    chart_json = _dumps({k: v for k, v in charts.items() if not k.startswith("_")})

    hero = f"""
    <div id="overview" class="hero">
      <div class="hero-block"><div class="hero-label">Portfolio NAV</div><div class="hero-val green">{_fc(metrics["current_nav"],0)}</div><div class="hero-sub">from {_fc(init_aum,0)} · {metrics["sessions"]} sessions</div></div>
      <div class="hero-block"><div class="hero-label">Total P&amp;L</div><div class="hero-val {'green' if metrics['total_pnl'] >= 0 else 'red'}">{_fc(metrics["total_pnl"],0)}</div><div class="hero-sub">since {CFG["inception"]}</div></div>
      <div class="hero-block"><div class="hero-label">Total Return</div><div class="hero-val {'green' if metrics['total_return'] >= 0 else 'red'}">{_fp(metrics["total_return"]*100)}</div><div class="hero-sub">{CFG["benchmark"]}: {_fp(metrics["bench_total_return"]*100)} · excess: {_fp(metrics["alpha"]*100)}</div></div>
      <div class="hero-block"><div class="hero-label">Sharpe / Sortino</div><div class="hero-val blue">{metrics["sharpe"]:.3f}</div><div class="hero-sub">Sortino: {metrics["sortino"]:.3f} | Calmar: {metrics["calmar"]:.3f}</div></div>
      <div class="hero-right">
        <div class="hero-pill"><span class="pill-label">Beta</span><span class="pill-val" style="color:var(--amber);">{metrics["beta"]:.2f}x</span></div>
        <div class="hero-pill"><span class="pill-label">Max DD</span><span class="pill-val" style="color:var(--red);">{_fp(metrics["mdd"]*100)}</span></div>
        <div class="hero-pill"><span class="pill-label">Vol (ann.)</span><span class="pill-val" style="color:var(--blue);">{_fpu(metrics["vol"]*100)}</span></div>
        <div class="hero-pill"><span class="pill-label">Hit Ratio</span><span class="pill-val" style="color:var(--green);">{_fpu(metrics["hit"]*100)}</span></div>
      </div>
    </div>
    """

    kpi_grid = "".join([
        _kpi("Ann. Return", _fp(metrics["ann_return"]*100), f"QQQ: {_fp(metrics['ann_bench']*100)}", "g"),
        _kpi("Jensen's Alpha", _fp(metrics["jalpha"]*100), f"after β · M² {_fp(metrics['m2']*100,1)}", "b"),
        _kpi("Beta vs QQQ", f"{metrics['beta']:.2f}x", f"ρ={metrics['corr']:.3f}", "a"),
        _kpi("VaR 95% (1d)", _fp(metrics["var95"]*100), f"CVaR: {_fp(metrics['cvar95']*100)}", "r"),
        _kpi("Info Ratio", f"{metrics['ir']:.3f}", f"TE: {_fpu(metrics['te']*100)}", "p"),
    ])

    intelligence_html = "".join(f"<div class='intel-section'><div class='intel-head'>{_esc(t)}</div><div class='intel-body'>{_esc(b)}</div></div>" for t, b in intel)

    links = []
    if CFG.get("github_url"):
        links.append(f"<a href='{_esc(CFG['github_url'])}' target='_blank' rel='noopener'>GitHub</a>")
    if CFG.get("linkedin_url"):
        links.append(f"<a href='{_esc(CFG['linkedin_url'])}' target='_blank' rel='noopener'>LinkedIn</a>")
    if CFG.get("contact_email"):
        links.append(f"<a href='mailto:{_esc(CFG['contact_email'])}'>Contact</a>")
    links_html = " · ".join(links)

    about_html = _sections_html(content["about"])
    attribution_block = ""
    if at is not None:
        attribution_block = f"""
<div id="attribution" class="section-label"><span class="slnum">02</span> Is it just beta? — Performance attribution</div>
<div class="row r65">
  <div class="card"><div class="card-head"><span class="card-title">Portfolio vs QQQ levered to the same beta</span><span class="card-badge">β-matched</span></div><div class="chart-wrap" id="beta_matched"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Reading the numbers</span><span class="card-badge">auto-updated</span></div><div class="card-body prose">{_attribution_text(metrics, at)}</div></div>
</div>
<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">CAPM decomposition</span></div><div class="chart-wrap" id="capm"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Rolling Jensen's alpha</span><span class="card-badge">{CFG["alpha_w"]}d window</span></div><div class="chart-wrap" id="rolling_alpha"></div></div>
</div>
<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">Selection vs beta, per position</span><span class="card-badge">single-period CAPM</span></div><div class="chart-wrap" id="selection"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Risk budget</span><span class="card-badge">ex-post covariance</span></div><div class="chart-wrap" id="risk_contrib"></div>
    <div class="card-body prose" style="padding-top:0">{_risk_text(metrics, at, structure)}</div>
    <div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Bucket</th><th class="num">N</th><th class="num">Capital</th><th class="num">Risk share</th><th class="num">P&amp;L</th><th class="num">Selection P&amp;L</th></tr></thead><tbody>{_bucket_table(at["buckets"])}</tbody></table></div>
  </div>
</div>
"""

    theses_block = ""
    if content["theses"]:
        theses_block = f"""
<div id="theses" class="section-label"><span class="slnum">05</span> Investment theses</div>
<div class="card" style="margin-bottom:16px;"><div class="card-body">{_thesis_cards(content["theses"], pos, at["sel"] if at is not None else pd.DataFrame(columns=["ticker","selection_pnl"]))}</div></div>
"""

    journal_block = ""
    if content["journal"]:
        journal_block = f"""
<div id="journal" class="section-label"><span class="slnum">06</span> Portfolio journal</div>
<div class="card" style="margin-bottom:16px;"><div class="card-body">
  <div class="prose lead"><p>Market notes at the key moments of the portfolio's life: what happened, why the fund moved the way it did against {CFG["benchmark"]}, and what I expected next. 
  The figures in each note (fund vs benchmark, beta, drawdown over the period) are computed automatically from the price data. Numbered markers on the performance chart point to each note.</p></div>
  {_journal_html(content["journal"])}
</div></div>
"""

    outlook_block = ""
    if content["watchlist"] or content["multi_asset"]:
        outlook_block = f"""
<div id="outlook" class="section-label"><span class="slnum">07</span> Watchlist &amp; multi-asset view</div>
<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">Watchlist — names I would add</span></div><div class="card-body">{_sections_html(content["watchlist"])}</div></div>
  <div class="card"><div class="card-head"><span class="card-title">If this were a multi-asset portfolio</span></div><div class="card-body">{_sections_html(content["multi_asset"])}</div></div>
</div>
"""

    methodology_block = f"""
<div id="methodology" class="section-label"><span class="slnum">09</span> Methodology &amp; disclaimer</div>
<div class="card" style="margin-bottom:16px;"><div class="card-body prose cols">
<h3>Construction</h3>
<p>Buy-and-hold book of {len(pos)} US equities opened on {CFG["inception"]} with ~{_fc(init_aum,0)} of capital at the entry prices listed in <code>holdings.csv</code>. No rebalancing, no leverage, no shorting. Benchmarks ({CFG["benchmark"]}, {CFG["bench2"]}) are bought with the same capital on the first session. Prices are daily closes from Yahoo Finance, refreshed three times per US trading day by GitHub Actions.</p>
<h3>Returns &amp; risk</h3>
<p>Total return and P&amp;L are measured against capital invested at entry prices. Annualisation uses {CFG["af"]} sessions and a {CFG["rf"]*100:.1f}% risk-free rate. Beta and correlation are estimated on daily returns over the full period; rolling statistics use {CFG["roll_w"]}-day windows ({CFG["alpha_w"]} days for alpha). VaR/CVaR are historical, one-day, 95%.</p>
<h3>Attribution</h3>
<p>Jensen's alpha = annualised return − [rf + β × (benchmark return − rf)]. The β-matched line re-levers daily {CFG["benchmark"]} excess returns by the portfolio beta. Per-position selection P&amp;L = actual P&amp;L − cost × [rf + βᵢ × ({CFG["benchmark"]} return − rf)]. Risk contribution uses current weights and the full-period covariance matrix (RCᵢ = wᵢ(Σw)ᵢ / w′Σw). M² = rf + Sharpe × σ(benchmark).</p>
<h3>Journal</h3>
<p>Notes dated before {CFG.get("journal_written","2026-11-23")} are a retrospective analysis written on that date with the information available at the time of each event; they are labelled "written …". Later notes are written live.</p>
<h3>Data</h3>
<p>Raw data: <a href="data/daily_history.csv">daily history (CSV)</a> · <a href="data/monthly_returns.csv">monthly returns (CSV)</a>.</p>
<h3>Disclaimer</h3>
<p>Academic, paper-traded portfolio created as a personal project. Not investment advice, not an offer of any financial product. Past performance does not predict future results.</p>
</div></div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_esc(CFG["portfolio_name"])} — Portfolio Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{--bg:#07090F;--panel:#0C1018;--card:#111520;--card2:#161C2A;--border:#1C2840;--border2:#243350;--green:#00D97E;--blue:#2D7DD2;--red:#E8304A;--amber:#F0A500;--purple:#7C5CBF;--cyan:#00B4D8;--text:#D8E2F0;--sub:#7A8FAD;--dim:#3A4A65;--mono:'JetBrains Mono','Courier New',monospace;--sans:'DM Sans',system-ui,sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}} html{{scroll-behavior:smooth}} body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.5;min-height:100vh;overflow-x:hidden}}
.topbar{{position:sticky;top:0;z-index:200;background:rgba(7,9,15,.97);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 32px;height:52px}}
.logo{{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--green)}} .logo sup{{font-size:9px;color:var(--sub);font-weight:400;margin-left:6px}}
.topbar-nav{{display:flex;gap:2px}} .topbar-nav a{{color:var(--sub);text-decoration:none;font-size:10.5px;letter-spacing:.8px;text-transform:uppercase;padding:5px 12px;border-radius:4px;font-family:var(--mono)}}
.topbar-nav a:hover,.topbar-nav a.active{{color:var(--text);background:var(--card2)}} .updated{{font-family:var(--mono);font-size:10px;color:var(--sub)}}
main{{max-width:1500px;margin:0 auto;padding:24px 32px 60px}} .section-label{{font-family:var(--mono);font-size:9px;font-weight:700;color:var(--sub);letter-spacing:3px;text-transform:uppercase;margin:32px 0 16px;display:flex;align-items:center;gap:12px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}} .section-label .slnum{{color:var(--dim);font-size:8px}}
.row{{display:grid;gap:16px;margin-bottom:16px}} .r2{{grid-template-columns:1fr 1fr}} .r3{{grid-template-columns:1fr 1fr 1fr}} .r65{{grid-template-columns:3fr 2fr}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}} .card-head{{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--border)}}
.card-title{{font-family:var(--mono);font-size:9px;font-weight:700;color:var(--sub);letter-spacing:2px;text-transform:uppercase}} .card-badge{{font-family:var(--mono);font-size:8px;color:var(--dim);padding:2px 8px;background:var(--card2);border-radius:3px;border:1px solid var(--border)}}
.card-body{{padding:16px 18px}} .chart-wrap{{padding:4px 6px 8px}}
.hero{{display:grid;grid-template-columns:auto auto auto auto 1fr;border:1px solid var(--border);border-radius:10px;background:var(--card);overflow:hidden;margin-bottom:16px}} .hero-block{{padding:20px 28px;border-right:1px solid var(--border)}}
.hero-label{{font-family:var(--mono);font-size:8.5px;color:var(--sub);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px}} .hero-val{{font-family:var(--mono);font-size:28px;font-weight:700;line-height:1}}
.hero-val.green{{color:var(--green)}} .hero-val.red{{color:var(--red)}} .hero-val.blue{{color:var(--blue)}} .hero-sub{{font-family:var(--mono);font-size:10px;color:var(--sub);margin-top:6px}}
.hero-right{{padding:16px 28px;display:flex;flex-direction:column;justify-content:center;gap:10px}} .hero-pill{{display:flex;align-items:center;gap:10px}} .pill-label{{font-family:var(--mono);font-size:9px;color:var(--sub);width:90px;text-transform:uppercase}} .pill-val{{font-family:var(--mono);font-size:11px;font-weight:600}}
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}} .kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 16px;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}} .kpi.g::before{{background:var(--green)}} .kpi.b::before{{background:var(--blue)}} .kpi.r::before{{background:var(--red)}} .kpi.a::before{{background:var(--amber)}} .kpi.p::before{{background:var(--purple)}}
.kpi-label{{font-family:var(--mono);font-size:8.5px;color:var(--sub);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}} .kpi-val{{font-family:var(--mono);font-size:20px;font-weight:700;line-height:1}} .kpi-sub{{font-family:var(--mono);font-size:9px;color:var(--sub);margin-top:5px}}
.bucket{{font-family:var(--mono);font-size:8px;padding:1px 7px;border-radius:3px;text-transform:uppercase}} .bucket.core{{background:rgba(45,125,210,.12);color:var(--blue)}} .bucket.growth{{background:rgba(0,217,126,.10);color:var(--green)}} .bucket.speculative{{background:rgba(240,165,0,.10);color:var(--amber)}}
.data-table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}} .data-table th{{font-size:8.5px;color:var(--sub);letter-spacing:1.5px;text-transform:uppercase;padding:10px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--card2);font-weight:600;white-space:nowrap}}
.data-table td{{padding:9px 12px;border-bottom:1px solid rgba(28,40,64,.5);white-space:nowrap}} .data-table tr:last-child td{{border-bottom:none}} .data-table tr:hover td{{background:rgba(45,125,210,.04)}} .num{{text-align:right}}
.td-ticker{{font-weight:700;color:var(--text);font-size:12px}} .td-name{{color:var(--sub);font-size:10.5px;max-width:240px;overflow:hidden;text-overflow:ellipsis}}
.intel-section{{margin-bottom:20px}} .intel-head{{font-family:var(--mono);font-size:8.5px;font-weight:700;color:var(--blue);letter-spacing:2.5px;text-transform:uppercase;padding-bottom:8px;border-bottom:1px solid var(--border);margin-bottom:12px}} .intel-body{{font-size:12.5px;line-height:1.9;color:var(--text)}}
footer{{border-top:1px solid var(--border);padding:24px 32px;display:flex;align-items:center;justify-content:space-between;max-width:1500px;margin:0 auto}} .footer-left,.footer-right{{font-family:var(--mono);font-size:10px;color:var(--dim)}}
.prose{{font-size:13px;line-height:1.75;color:var(--text)}} .prose p{{margin:0 0 10px}} .prose ul,.prose ol{{margin:0 0 10px 18px}} .prose li{{margin-bottom:4px}} .prose b,.prose strong{{color:#fff}}
.prose h3{{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--blue);margin:14px 0 8px}} .prose h3:first-child{{margin-top:0}} .prose h4{{font-size:12px;color:var(--amber);margin:12px 0 6px}}
.prose code{{font-family:var(--mono);font-size:11px;background:var(--card2);padding:1px 5px;border-radius:3px}} .prose a{{color:var(--blue)}} .prose.lead{{color:var(--sub);margin-bottom:14px}} .prose.cols{{columns:2 420px;column-gap:40px}} .prose.cols h3{{break-after:avoid}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}} .chip{{font-family:var(--mono);font-size:9.5px;color:var(--sub);background:var(--card2);border:1px solid var(--border);padding:2px 8px;border-radius:3px}}
.thesis-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .thesis{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px}} .thesis summary{{cursor:pointer;list-style:none}} .thesis summary::-webkit-details-marker{{display:none}}
.thesis summary .th-title{{font-family:var(--mono);font-size:12px;font-weight:700}} .thesis summary .th-title::before{{content:'+ ';color:var(--dim)}} .thesis[open] summary .th-title::before{{content:'– '}} .thesis .prose{{margin-top:10px;border-top:1px solid var(--border);padding-top:10px}}
.timeline{{position:relative;margin-left:14px;border-left:1px solid var(--border2);padding-left:28px}} .jentry{{position:relative;margin-bottom:28px}} .jdot{{position:absolute;left:-43px;top:0;width:28px;height:28px;border-radius:50%;background:var(--amber);color:#07090F;font-family:var(--mono);font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center}}
.jhead{{display:flex;gap:10px;align-items:center;font-family:var(--mono);font-size:10px;color:var(--sub)}} .jdate{{color:var(--text);font-weight:600}} .jtag{{text-transform:uppercase;letter-spacing:1px;font-size:8.5px;padding:1px 6px;border-radius:3px;background:var(--card2)}}
.jtag.rally{{color:var(--green)}} .jtag.drawdown{{color:var(--red)}} .jtag.consolidation{{color:var(--amber)}} .jtag.outlook{{color:var(--cyan)}} .jtag.review{{color:var(--purple)}} .written{{color:var(--dim);font-style:italic}}
.jentry h3{{font-size:15px;margin:6px 0 2px;font-weight:600}}
.contact a{{color:var(--sub);text-decoration:none;font-family:var(--mono);font-size:10px}} .contact a:hover{{color:var(--text)}}
.intro{{display:grid;grid-template-columns:1fr;gap:0;margin-bottom:16px}}
@media (max-width:1200px){{.hero{{grid-template-columns:1fr 1fr}}.kpi-grid{{grid-template-columns:repeat(3,1fr)}}.topbar-nav a{{padding:5px 8px}}}}
@media (max-width:900px){{.topbar{{flex-wrap:wrap;height:auto;padding:8px 16px;gap:6px}}.topbar>div:first-child{{flex-wrap:wrap;gap:8px!important;width:100%;min-width:0}}.topbar-nav{{overflow-x:auto;width:100%;min-width:0;scrollbar-width:none}}.topbar-nav a{{white-space:nowrap}}.updated{{display:none}}.thesis-grid{{grid-template-columns:1fr}}}}
@media (max-width:768px){{main{{padding:16px}}.hero{{grid-template-columns:1fr 1fr}}.hero-block{{padding:16px}}.hero-val{{font-size:22px}}.kpi-grid{{grid-template-columns:1fr 1fr}}.r2,.r3,.r65{{grid-template-columns:1fr}}footer{{flex-direction:column;gap:8px;text-align:center}}}}
@media (max-width:480px){{.hero{{grid-template-columns:1fr}}.hero-block{{border-right:none;border-bottom:1px solid var(--border)}}.timeline{{margin-left:6px;padding-left:20px}}.jdot{{left:-35px;width:24px;height:24px}}}}
</style>
</head>
<body>
<nav class="topbar">
  <div style="display:flex;align-items:center;gap:28px;">
    <div class="logo">{_esc(CFG["portfolio_name"])} <sup>{_esc(CFG["school"])}</sup></div>
    <nav class="topbar-nav">
      <a href="#overview" class="active">Overview</a>
      <a href="#performance">Performance</a>
      <a href="#attribution">Attribution</a>
      <a href="#risk">Risk</a>
      <a href="#positions">Positions</a>
      <a href="#theses">Theses</a>
      <a href="#journal">Journal</a>
      <a href="#outlook">Outlook</a>
      <a href="#methodology">Method</a>
    </nav>
  </div>
  <div class="updated"><span class="contact">{links_html}</span>{" · " if links_html else ""}Updated: {now_utc}</div>
</nav>
<main>
{hero}
<div class="kpi-grid">{kpi_grid}</div>
{f'<div class="card" style="margin-bottom:16px;"><div class="card-head"><span class="card-title">About the fund</span><span class="card-badge">mandate &amp; process</span></div><div class="card-body prose cols">{about_html}</div></div>' if about_html else ''}

<div id="performance" class="section-label"><span class="slnum">01</span> Performance</div>
<div class="card" style="margin-bottom:16px;"><div class="card-head"><span class="card-title">Portfolio vs Benchmarks — Base 100 from {CFG["inception"]}</span><span class="card-badge">NAV-weighted</span></div><div class="chart-wrap" id="perf"></div></div>

<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">Drawdown from Peak</span><span class="card-badge">vs {CFG["benchmark"]}</span></div><div class="chart-wrap" id="drawdown"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Monthly Returns vs {CFG["benchmark"]}</span></div><div class="chart-wrap" id="monthly_returns"></div></div>
</div>

{attribution_block}
<div id="risk" class="section-label"><span class="slnum">03</span> Risk Analytics</div>
<div class="row r3">
  <div class="card"><div class="card-head"><span class="card-title">Rolling Volatility</span></div><div class="chart-wrap" id="rolling_vol"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Rolling Beta</span></div><div class="chart-wrap" id="rolling_beta"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Rolling Sharpe</span></div><div class="chart-wrap" id="rolling_sharpe"></div></div>
</div>

<div class="card" style="margin-bottom:16px;">
  <div class="card-head"><span class="card-title">Complete Risk / Return Metrics</span><span class="card-badge">rf={CFG["rf"]*100:.2f}%</span></div>
  <div class="card-body" style="padding:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Metric</th><th class="num">Value</th><th>Comment</th></tr></thead><tbody>{_metrics_table(metrics)}</tbody></table></div>
</div>

<div id="positions" class="section-label"><span class="slnum">04</span> Positions</div>
<div class="row r3">
  <div class="card"><div class="card-head"><span class="card-title">Top Weights</span></div><div class="chart-wrap" id="top_weights"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Sector Allocation</span></div><div class="chart-wrap" id="sector_alloc"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Thematic Allocation</span></div><div class="chart-wrap" id="theme_alloc"></div></div>
</div>

<div class="row r2" style="margin-bottom:16px;">
  <div class="card"><div class="card-head"><span class="card-title">Position P&amp;L Attribution</span></div><div class="chart-wrap" id="pnl_attr"></div></div>
  <div class="card"><div class="card-head"><span class="card-title">Monthly Heatmap</span></div><div class="chart-wrap" id="heatmap"></div></div>
</div>

<div class="card" style="margin-bottom:16px;">
  <div class="card-head"><span class="card-title">Holdings — {len(pos)} Positions · NAV {_fc(metrics["current_nav"],0)}</span><span class="card-badge">Buy &amp; Hold</span></div>
  <div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Risk</th><th class="num">Qty</th><th class="num">Buy</th><th class="num">Price</th><th class="num">Value</th><th class="num">P&amp;L</th><th class="num">Return</th><th class="num">Weight</th><th class="num">Day</th><th class="num">1W</th><th class="num">1M</th><th class="num">Beta</th></tr></thead><tbody>{_positions_table(pos)}</tbody></table></div>
</div>

<div class="row r65">
  <div class="card"><div class="card-head"><span class="card-title">Daily Ledger</span></div><div class="card-body" style="padding:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Date</th><th class="num">NAV</th><th class="num">Daily P&amp;L</th><th class="num">Portfolio</th><th class="num">QQQ</th><th class="num">Active</th><th class="num">Drawdown</th></tr></thead><tbody>{_ledger_table(ledger)}</tbody></table></div></div>
  <div id="intelligence" class="card"><div class="card-head"><span class="card-title">Portfolio Intelligence</span></div><div class="card-body">{intelligence_html}</div></div>
</div>

{theses_block}
{journal_block}
{outlook_block}
<div id="stress" class="section-label"><span class="slnum">08</span> Stress &amp; Scenario</div>
<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">Stress Tests</span></div><div class="chart-wrap" id="stress"></div><div class="card-body" style="padding-top:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Scenario</th><th>Description</th><th class="num">P&amp;L Impact</th><th class="num">Return Impact</th><th class="num">NAV After</th></tr></thead><tbody>{_stress_table(stress)}</tbody></table></div></div>
  <div class="card"><div class="card-head"><span class="card-title">12M Scenario Envelope</span></div><div class="chart-wrap" id="forecast"></div><div class="card-body" style="padding-top:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Horizon</th><th class="num">Start NAV</th><th class="num">P05</th><th class="num">P25</th><th class="num">Median</th><th class="num">P75</th><th class="num">P95</th></tr></thead><tbody>{_forecast_table(fcast_sum)}</tbody></table></div></div>
</div>

<div class="section-label"><span class="slnum">08b</span> News &amp; Monitoring</div>
<div class="row r2">
  <div class="card"><div class="card-head"><span class="card-title">Sector Breakdown</span></div><div class="card-body" style="padding:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Sector</th><th class="num">Weight</th><th class="num">Market Value</th><th class="num">P&amp;L</th><th class="num">N</th></tr></thead><tbody>{_sector_table(structure)}</tbody></table></div></div>
  <div class="card"><div class="card-head"><span class="card-title">News Flow</span></div><div class="card-body" style="padding:0;overflow-x:auto;"><table class="data-table"><thead><tr><th>Ticker</th><th>Headline</th><th>Source</th><th>Published</th></tr></thead><tbody>{_news_table(news)}</tbody></table></div></div>
</div>
{methodology_block}
</main>
<footer><div class="footer-left">{_esc(CFG["portfolio_name"])} · managed by {_esc(CFG["manager"])} · academic paper portfolio, not investment advice</div><div class="footer-right contact">{links_html}{" · " if links_html else ""}Built with Python, pandas &amp; Plotly · auto-refreshed by GitHub Actions</div></footer>
<script>
const CHARTS = {chart_json};
const CONFIG = {{ displayModeBar: false, responsive: true, scrollZoom: false }};
for (const [id, fig] of Object.entries(CHARTS)) {{
  const el = document.getElementById(id);
  if (el) Plotly.newPlot(id, fig.data, fig.layout, CONFIG);
}}
</script>
</body>
</html>"""


def build_snapshot(metrics: dict, pos: pd.DataFrame, structure: dict, ledger: pd.DataFrame,
                   stress: pd.DataFrame, fcast_sum: pd.DataFrame, news: pd.DataFrame,
                   intel: list) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": CFG,
        "metrics": {
            "current_nav": metrics["current_nav"],
            "daily_pnl": metrics["daily_pnl"],
            "daily_return": metrics["daily_return"],
            "total_pnl": metrics["total_pnl"],
            "total_return": metrics["total_return"],
            "bench_total_return": metrics["bench_total_return"],
            "alpha": metrics["alpha"],
            "ann_return": metrics["ann_return"],
            "ann_bench": metrics["ann_bench"],
            "vol": metrics["vol"],
            "bvol": metrics["bvol"],
            "ddev": metrics["ddev"],
            "bddev": metrics["bddev"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "beta": metrics["beta"],
            "corr": metrics["corr"],
            "jalpha": metrics["jalpha"],
            "te": metrics["te"],
            "ir": metrics["ir"],
            "mdd": metrics["mdd"],
            "bmdd": metrics["bmdd"],
            "calmar": metrics["calmar"],
            "var95": metrics["var95"],
            "cvar95": metrics["cvar95"],
            "skew": metrics["skew"],
            "kurt": metrics["kurt"],
            "treynor": metrics["treynor"],
            "omega": metrics["omega"],
            "upc": metrics["upc"],
            "dnc": metrics["dnc"],
            "hit": metrics["hit"],
            "bsharpe": metrics["bsharpe"],
            "m2": metrics["m2"],
            "sessions": metrics["sessions"],
        },
        "positions": pos.to_dict(orient="records"),
        "structure": {
            "sector": structure["sector"].to_dict(orient="records"),
            "theme": structure["theme"].to_dict(orient="records"),
            "bucket": structure["bucket"].to_dict(orient="records"),
            "hhi": structure["hhi"],
            "eff_n": structure["eff_n"],
            "top5": structure["top5"],
        },
        "ledger": ledger.to_dict(orient="records"),
        "stress": stress.to_dict(orient="records"),
        "forecast_summary": fcast_sum.to_dict(orient="records"),
        "news": news.to_dict(orient="records") if not news.empty else [],
        "intelligence": [{"title": t, "body": b} for t, b in intel],
    }


def export_history(frame: dict, metrics: dict, folder: Path) -> None:
    """Publish the daily and monthly series as CSV next to the dashboard (transparency + reuse)."""
    folder.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame({
        "nav": frame["nav"].round(2),
        "fund_base100": frame["b100"].round(3),
        f"{CFG['benchmark'].lower()}_base100": frame["bb100"].round(3),
        f"{CFG['bench2'].lower()}_base100": frame["b2b100"].round(3) if not frame["b2b100"].empty else np.nan,
        "fund_drawdown_pct": frame["dd"].round(3),
    })
    daily.index.name = "date"
    daily.to_csv(folder / "daily_history.csv")
    monthly = pd.DataFrame({"fund_return_pct": (metrics["mp"] * 100).round(2), f"{CFG['benchmark'].lower()}_return_pct": (metrics["mb"] * 100).round(2)})
    monthly["active_pct"] = (monthly.iloc[:, 0] - monthly.iloc[:, 1]).round(2)
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    monthly.to_csv(folder / "monthly_returns.csv")


def main():
    holdings = load_holdings(ROOT / "holdings.csv")
    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["bench2"]]
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prices = download_prices(tickers, CFG["inception"], end)
    frame = build_frame(prices, holdings)
    metrics = compute_metrics(frame)
    pos = compute_positions(frame, holdings)
    structure = compute_structure(pos)
    ledger = build_ledger(frame)
    heatmap = build_heatmap(metrics["mp"])
    stress = build_stress(pos, metrics)
    fcast_sum, fcast_paths = build_forecast(frame)
    news = build_news(holdings)
    intel = build_intelligence(metrics, pos, structure, stress)
    charts = make_charts(frame, metrics, pos, structure, heatmap, stress, fcast_paths)

    content = load_content(ROOT)
    content["journal"] = journal_with_stats(content["journal"], frame)
    at = build_attribution(frame, metrics, pos)
    extra = make_attr_charts(frame, metrics, at, content["journal"])
    charts["perf"] = _add_journal_marks(charts["perf"], extra.pop("_journal_marks"))
    charts.update(extra)

    html = generate_html(holdings, metrics, pos, structure, ledger, stress, fcast_sum, news, intel, charts, at=at, content=content)
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")

    export_history(frame, metrics, OUT_HTML.parent / "data")

    snapshot = build_snapshot(metrics, pos, structure, ledger, stress, fcast_sum, news, intel)
    OUT_JSON.write_text(_dumps(snapshot), encoding="utf-8", newline="\n")

    print(f"Generated {OUT_HTML}")
    print(f"Generated {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Build failed: {exc}")
        traceback.print_exc()
        raise SystemExit(1)  # non-zero exit so the workflow's retry loop kicks in
