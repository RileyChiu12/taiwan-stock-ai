# -*- coding: utf-8 -*-
"""
advanced.py - 單檔深度分析的進階圖表

依 Colab 筆記本 STEP 8 / 9 / 10 / 11 / 12 / 14 整合：
  布林通道（20/60 日 + 收縮/放大區間）、ATR 波動度、季度營收與毛利率、
  三大法人（當日/累計 + 分價量 POC）、融資融券、外資台指期與借券賣出。

每張圖獨立計算；任何一張失敗只回傳錯誤訊息，不影響其他圖與主分析。
籌碼資料取不到時直接顯示原因，不以模擬數據替代。
"""
import os
import json

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
UP, DOWN = "#E53935", "#2E7D32"          # 台股慣例：紅漲綠跌
DISPLAY_DAYS = {"3mo": 92, "6mo": 183, "1y": 365, "2y": 730, "3y": 1095, "5y": 1826}
# 指標需要的暖身資料（BB60 + 100 日百分位），下載期間至少比顯示期間多約 1 年
DOWNLOAD_PERIOD = {"3mo": "2y", "6mo": "2y", "1y": "2y", "2y": "3y", "3y": "5y", "5y": "10y", "max": "max"}
MAX_BARS = 750                            # 長區間只畫最近 750 根，避免頁面過重


# ───────────────────────── 共用工具 ─────────────────────────
def _fig_json(fig):
    return json.loads(fig.to_json())


def _clean_ohlcv(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def download_history(ticker, period):
    """上市 .TW 抓不到時改抓上櫃 .TWO。回傳 (df, yahoo_symbol)。"""
    for suffix in (".TW", ".TWO"):
        sym = f"{ticker}{suffix}"
        try:
            df = _clean_ohlcv(yf.download(sym, period=period, progress=False, auto_adjust=False))
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            return df, sym
    return pd.DataFrame(), None


def _display_slice(df, period):
    if period in DISPLAY_DAYS:
        start = df.index.max() - pd.Timedelta(days=DISPLAY_DAYS[period])
        d = df.loc[df.index >= start]
    else:
        d = df
    return d.tail(MAX_BARS)


def _rangebreaks(index):
    """隱藏非交易日，讓 K 線不出現週末與假日空白。"""
    if len(index) < 2:
        return []
    missing = pd.date_range(index.min(), index.max()).difference(index)
    return [dict(values=missing.strftime("%Y-%m-%d").tolist())]


def _spans(mask):
    """把布林遮罩轉成連續區間 [(start, end), ...]。"""
    spans, start, prev = [], None, None
    for dt, flag in mask.items():
        if flag and start is None:
            start = dt
        elif not flag and start is not None:
            spans.append((start, prev))
            start = None
        prev = dt
    if start is not None:
        spans.append((start, prev))
    return spans


def _candles(d, name="K線"):
    return go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                          name=name, increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False)


def _base_layout(fig, title, height, index):
    fig.update_layout(title=dict(text=title, y=0.995, yanchor="top"), template="plotly_white", height=height,
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.035, x=0),
                      margin=dict(l=60, r=60, t=105, b=40))
    fig.update_xaxes(rangeslider_visible=False)
    return _rangebreaks(index)


def finmind_fetch(dataset, data_id, start, end):
    """回傳 (DataFrame, 錯誤訊息)。Token 以 Bearer Header 傳遞。"""
    token = (os.environ.get("FINMIND_TOKEN") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        return pd.DataFrame(), "未設定 FINMIND_TOKEN"
    try:
        r = requests.get(FINMIND_URL, headers={"Authorization": f"Bearer {token}"}, timeout=30,
                         params={"dataset": dataset, "data_id": data_id, "start_date": start, "end_date": end})
        js = r.json()
    except Exception as e:
        return pd.DataFrame(), f"{dataset} 連線失敗：{e}"
    if r.status_code != 200 or js.get("status") != 200:
        return pd.DataFrame(), f"{dataset} 取得失敗：HTTP {r.status_code}｜{js.get('msg')}"
    if not js.get("data"):
        return pd.DataFrame(), f"{dataset} 回傳 0 筆（日期區間內無資料，或方案無此權限）"
    out = pd.DataFrame(js["data"])
    out["date"] = pd.to_datetime(out["date"])
    return out, None


def _pct_rank(series, value):
    s = series.dropna()
    return float((s <= value).mean() * 100) if len(s) else np.nan


# ───────────────────────── STEP 8 布林通道 ─────────────────────────
def bollinger(full, period, ticker):
    df = full.copy()
    c = df["close"]
    df["bb20_mid"] = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    df["bb20_up"], df["bb20_low"] = df["bb20_mid"] + 2 * sd20, df["bb20_mid"] - 2 * sd20
    df["bw20"] = (df["bb20_up"] - df["bb20_low"]) / df["bb20_mid"]
    df["bb60_mid"] = c.rolling(60).mean()
    sd60 = c.rolling(60).std()
    df["bb60_up"], df["bb60_low"] = df["bb60_mid"] + 2 * sd60, df["bb60_mid"] - 2 * sd60
    lo_th = df["bw20"].rolling(100).quantile(0.15)
    hi_th = df["bw20"].rolling(100).quantile(0.85)
    squeeze = df["bw20"] < lo_th
    expansion = (df["bw20"] > hi_th) & (df["bw20"].diff() > 0)

    d = _display_slice(df, period)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    fig.add_trace(go.Scatter(x=d.index, y=d["bb60_low"], line=dict(color="rgba(255,165,0,.35)", width=1), name="60日下軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb60_up"], line=dict(color="rgba(255,165,0,.35)", width=1),
                             fill="tonexty", fillcolor="rgba(255,165,0,.05)", name="60日上軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb20_low"], line=dict(color="rgba(233,30,99,.35)", width=1), name="20日下軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb20_up"], line=dict(color="rgba(233,30,99,.35)", width=1),
                             fill="tonexty", fillcolor="rgba(233,30,99,.06)", name="20日上軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb60_mid"], line=dict(color="rgb(255,140,0)", width=2.5), name="60日中軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb20_mid"], line=dict(color="rgb(233,30,99)", width=2.5), name="20日中軌"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], line=dict(color="black", width=1.5), name="收盤價"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bw20"] * 100, line=dict(color="#455A64", width=1.5), name="20日帶寬 %"), 2, 1)

    for spans, color in [(_spans(squeeze.loc[d.index]), "rgba(0,123,255,.10)"),
                         (_spans(expansion.loc[d.index]), "rgba(255,193,7,.14)")]:
        for s, e in spans:
            fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0, layer="below")

    rb = _base_layout(fig, f"{ticker} 雙布林通道（🔵 收縮 Squeeze／🟡 放大 Expansion）", 720, d.index)
    fig.update_xaxes(rangebreaks=rb)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="帶寬 %", row=2, col=1)

    last = df.iloc[-1]
    pct_b = (last["close"] - last["bb20_low"]) / (last["bb20_up"] - last["bb20_low"]) * 100
    bw_rank = _pct_rank(df["bw20"].tail(250), last["bw20"])
    state = "🔵 收縮（低波動沉澱）" if squeeze.iloc[-1] else "🟡 放大（波動擴張）" if expansion.iloc[-1] else "⚪ 一般"
    pos = ("上軌之上" if last["close"] > last["bb20_up"] else "下軌之下" if last["close"] < last["bb20_low"]
           else "中軌之上" if last["close"] >= last["bb20_mid"] else "中軌之下")
    summary = "\n".join([
        f"- **收盤**：{last['close']:.2f}，位於 20 日通道 **{pos}**",
        f"- **20 日通道**：上軌 {last['bb20_up']:.2f}｜中軌 {last['bb20_mid']:.2f}｜下軌 {last['bb20_low']:.2f}",
        f"- **60 日中軌**：{last['bb60_mid']:.2f}",
        f"- **%B**：{pct_b:.1f}%（0% = 下軌，100% = 上軌）",
        f"- **20 日帶寬**：{last['bw20'] * 100:.2f}%，位於近一年 **{bw_rank:.0f}** 百分位",
        f"- **目前狀態**：{state}",
        "- 判定：帶寬低於近 100 日第 15 百分位為收縮；高於第 85 百分位且擴張中為放大。",
    ])
    return fig, summary


# ───────────────────────── STEP 12 ATR ─────────────────────────
def atr(full, period, ticker):
    df = full.copy()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(com=13, adjust=False).mean()           # Wilder 平滑
    df["atr_pct"] = df["atr14"] / df["close"] * 100

    d = _display_slice(df, period)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.68, 0.32],
                        specs=[[{}], [{"secondary_y": True}]])
    fig.add_trace(_candles(d), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["close"] - 2 * d["atr14"], line=dict(color="#6D4C41", width=1, dash="dot"),
                             name="收盤 − 2×ATR"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["atr14"], line=dict(color="#00838F", width=2), name="ATR(14) 元"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["atr_pct"], line=dict(color="#9E9E9E", width=1, dash="dot"),
                             name="ATR % 收盤"), 2, 1, secondary_y=True)
    rb = _base_layout(fig, f"{ticker} 14 日 ATR 真實波動幅度", 680, d.index)
    fig.update_xaxes(rangebreaks=rb)
    fig.update_yaxes(title_text="股價", row=1, col=1)
    fig.update_yaxes(title_text="ATR (元)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ATR %", row=2, col=1, secondary_y=True, showgrid=False)

    last = df.iloc[-1]
    rank = _pct_rank(df["atr_pct"].tail(250), last["atr_pct"])
    level = "高檔（震盪劇烈）" if rank >= 80 else "低檔（波動壓縮）" if rank <= 20 else "中間區域"
    summary = "\n".join([
        f"- **ATR(14)**：{last['atr14']:.2f} 元，約為收盤價的 **{last['atr_pct']:.2f}%**",
        f"- **波動位置**：近一年 **{rank:.0f}** 百分位 → {level}",
        f"- **2×ATR 距離**：{2 * last['atr14']:.2f} 元（收盤 − 2×ATR = {last['close'] - 2 * last['atr14']:.2f}）",
        "- ATR 高檔時價格震盪大，以 ATR 倍數設定停損距離需相應放寬；低檔時常伴隨盤整壓縮，留意之後的波動放大。",
    ])
    return fig, summary


# ───────────────────────── STEP 9 季度營收與毛利率 ─────────────────────────
def financials(yahoo_symbol, ticker):
    if not yahoo_symbol:
        raise RuntimeError("找不到 Yahoo Finance 代號")
    q = yf.Ticker(yahoo_symbol).quarterly_financials
    if q is None or q.empty:
        raise RuntimeError("Yahoo Finance 沒有此股票的季度損益表")
    q = q.T
    q.columns = [str(c).lower().strip() for c in q.columns]
    rev = next((c for c in q.columns if c in ("total revenue", "revenue", "operating revenue")), None)
    gp = next((c for c in q.columns if c == "gross profit"), None)
    if not rev or not gp:
        raise RuntimeError("季度損益表缺少營收或毛利欄位")
    f = q[[rev, gp]].apply(pd.to_numeric, errors="coerce").dropna()
    f.columns = ["revenue", "gross_profit"]
    f = f[f["revenue"] > 0].sort_index().tail(12)
    if f.empty:
        raise RuntimeError("季度營收資料為空")
    f["gm"] = f["gross_profit"] / f["revenue"] * 100
    labels = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in pd.to_datetime(f.index)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=f["revenue"] / 1e8, name="季度營收（億元）", marker_color="rgba(66,133,244,.75)"))
    fig.add_trace(go.Scatter(x=labels, y=f["gm"], name="毛利率 %", mode="lines+markers+text",
                             text=[f"{v:.1f}%" for v in f["gm"]], textposition="top center",
                             line=dict(color="rgb(219,68,85)", width=3)), secondary_y=True)
    fig.update_layout(title=f"{ticker} 季度營收與毛利率（資料來源：Yahoo Finance）", template="plotly_white",
                      height=520, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="營收（億元）", secondary_y=False)
    fig.update_yaxes(title_text="毛利率 %", secondary_y=True, showgrid=False,
                     range=[f["gm"].min() - 5, f["gm"].max() + 5])

    last = f.iloc[-1]
    lines = [f"- **最新一季（{labels[-1]}）**：營收 {last['revenue'] / 1e8:,.1f} 億元｜毛利率 **{last['gm']:.2f}%**"]
    if len(f) >= 2:
        prev = f.iloc[-2]
        lines.append(f"- **季增**：營收 {(last['revenue'] / prev['revenue'] - 1) * 100:+.1f}%｜毛利率 {last['gm'] - prev['gm']:+.2f} 個百分點")
    if len(f) >= 5:
        yoy = f.iloc[-5]
        lines.append(f"- **年增**：營收 {(last['revenue'] / yoy['revenue'] - 1) * 100:+.1f}%｜毛利率 {last['gm'] - yoy['gm']:+.2f} 個百分點")
    lines.append(f"- 共 {len(f)} 季資料；Yahoo Finance 台股季報通常只提供最近 4～6 季。")
    return fig, "\n".join(lines)


# ───────────────────────── STEP 10 三大法人 ─────────────────────────
def institutional(full, period, ticker):
    d = _display_slice(full, period)
    raw, err = finmind_fetch("TaiwanStockInstitutionalInvestorsBuySell", ticker,
                             d.index.min().strftime("%Y-%m-%d"), d.index.max().strftime("%Y-%m-%d"))
    if err:
        raise RuntimeError(err)
    raw["net"] = (pd.to_numeric(raw["buy"], errors="coerce") - pd.to_numeric(raw["sell"], errors="coerce")) / 1000
    groups = {"外資": ["Foreign_Investor", "Foreign_Dealer_Self"], "投信": ["Investment_Trust"],
              "自營商": ["Dealer_self", "Dealer_Hedging"]}
    piv = pd.DataFrame({k: raw[raw["name"].isin(v)].groupby("date")["net"].sum() for k, v in groups.items()})
    piv = piv.reindex(d.index).fillna(0)
    cum = piv.cumsum()

    # 分價量表與 POC（以收盤價分箱）
    bins = np.linspace(d["low"].min(), d["high"].max(), 51)
    mids = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.searchsorted(bins, d["close"].values) - 1, 0, 49)
    vbp = np.bincount(idx, weights=d["volume"].values, minlength=50)
    poc = float(mids[int(np.argmax(vbp))])

    specs = [[{}, {}]] + [[{"secondary_y": True}, None]] * 3
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, shared_yaxes=True, column_widths=[0.86, 0.14],
                        horizontal_spacing=0.01, vertical_spacing=0.04, row_heights=[0.40, 0.20, 0.20, 0.20],
                        specs=specs, subplot_titles=("股價／POC", "分價量", "外資", "投信", "自營商"))
    fig.add_trace(_candles(d), 1, 1)
    fig.add_trace(go.Scatter(x=[d.index.min(), d.index.max()], y=[poc, poc], mode="lines",
                             line=dict(color="red", width=1.5, dash="dash"), name=f"POC {poc:.1f}"), 1, 1)
    fig.add_trace(go.Bar(x=vbp / 1000, y=mids, orientation="h", marker_color="rgba(120,120,120,.35)",
                         name="分價量（張）", showlegend=False), 1, 2)
    line_colors = {"外資": "#1565C0", "投信": "#8E24AA", "自營商": "#E65100"}
    for row, k in zip((2, 3, 4), ("外資", "投信", "自營商")):
        fig.add_trace(go.Bar(x=d.index, y=piv[k], name=f"{k}當日（張）", opacity=.7,
                             marker_color=np.where(piv[k] >= 0, UP, DOWN), showlegend=False), row, 1)
        fig.add_trace(go.Scatter(x=d.index, y=cum[k], name=f"{k}累計", line=dict(color=line_colors[k], width=1.6)),
                      row, 1, secondary_y=True)
        fig.update_yaxes(title_text="當日張", row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="累計張", row=row, col=1, secondary_y=True, showgrid=False)
    rb = _base_layout(fig, f"{ticker} 三大法人當日買賣超與累計水位（附分價量 POC）", 1050, d.index)
    fig.update_xaxes(rangebreaks=rb, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)

    t, w = piv.iloc[-1], piv.tail(5).sum()
    last_date = d.index[-1].strftime("%Y/%m/%d")
    summary = "\n".join([
        f"**資料日期：{last_date}**（單位：張）",
        f"- **今日**：外資 {t['外資']:+,.0f}｜投信 {t['投信']:+,.0f}｜自營商 {t['自營商']:+,.0f}",
        f"- **近 5 日**：外資 {w['外資']:+,.0f}｜投信 {w['投信']:+,.0f}｜自營商 {w['自營商']:+,.0f}",
        f"- **區間累計**：外資 {cum['外資'].iloc[-1]:+,.0f}｜投信 {cum['投信'].iloc[-1]:+,.0f}｜自營商 {cum['自營商'].iloc[-1]:+,.0f}",
        f"- **POC 大量區**：{poc:.1f} 元（區間內收盤成交量最集中的價位），目前收盤 {d['close'].iloc[-1]:.2f}",
    ])
    return fig, summary


# ───────────────────────── STEP 11 融資融券 ─────────────────────────
def margin(full, period, ticker):
    d = _display_slice(full, period)
    raw, err = finmind_fetch("TaiwanStockMarginPurchaseShortSale", ticker,
                             d.index.min().strftime("%Y-%m-%d"), d.index.max().strftime("%Y-%m-%d"))
    if err:
        raise RuntimeError(err)
    m = raw.groupby("date")[["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]].last()
    m.columns = ["margin", "short"]
    m = m.apply(pd.to_numeric, errors="coerce").reindex(d.index).ffill()
    m["ratio"] = np.where(m["margin"] > 0, m["short"] / m["margin"] * 100, np.nan)
    m["margin_diff"], m["short_diff"], m["ratio_diff"] = m["margin"].diff(), m["short"].diff(), m["ratio"].diff()
    hi, lo = m["margin"].quantile(0.90), m["margin"].quantile(0.10)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.40, 0.20, 0.20, 0.20],
                        subplot_titles=("股價", "融資餘額（張）", "融券餘額（張）", "券資比 %"))
    fig.add_trace(_candles(d), 1, 1)
    fig.add_trace(go.Bar(x=d.index, y=m["margin"], name="融資餘額",
                         marker_color=np.where(m["margin_diff"].fillna(0) >= 0, UP, DOWN), showlegend=False), 2, 1)
    fig.add_trace(go.Bar(x=d.index, y=m["short"], name="融券餘額",
                         marker_color=np.where(m["short_diff"].fillna(0) >= 0, UP, DOWN), showlegend=False), 3, 1)
    fig.add_trace(go.Bar(x=d.index, y=m["ratio"], name="券資比",
                         marker_color=np.where(m["ratio_diff"].fillna(0) >= 0, UP, DOWN), showlegend=False), 4, 1)
    fig.add_hline(y=hi, line=dict(color="red", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=lo, line=dict(color="green", dash="dash", width=1), row=2, col=1)
    for spans, color in [(_spans(m["margin"] >= hi), "rgba(229,57,53,.10)"),
                         (_spans(m["margin"] <= lo), "rgba(46,125,50,.10)")]:
        for s, e in spans:
            fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0, layer="below", row=2, col=1)
    rb = _base_layout(fig, f"{ticker} 融資融券與券資比（紅增綠減）", 900, d.index)
    fig.update_xaxes(rangebreaks=rb)
    fig.update_layout(showlegend=False, margin=dict(t=70))

    last, w = m.iloc[-1], m.tail(6)
    summary = "\n".join([
        f"**資料日期：{d.index[-1].strftime('%Y/%m/%d')}**（單位：張）",
        f"- **融資餘額**：{last['margin']:,.0f}（今日 {last['margin_diff']:+,.0f}｜近 5 日 {w['margin'].iloc[-1] - w['margin'].iloc[0]:+,.0f}）",
        f"- **融券餘額**：{last['short']:,.0f}（今日 {last['short_diff']:+,.0f}｜近 5 日 {w['short'].iloc[-1] - w['short'].iloc[0]:+,.0f}）",
        f"- **券資比**：{last['ratio']:.2f}%（近 5 日 {w['ratio'].iloc[-1] - w['ratio'].iloc[0]:+.2f} 個百分點）",
        f"- **融資水位**：高水位門檻（90%）{hi:,.0f}｜低水位門檻（10%）{lo:,.0f} → 目前 "
        + ("**高水位區**" if last["margin"] >= hi else "**低水位區**" if last["margin"] <= lo else "中間區域"),
    ])
    return fig, summary


# ───────────────────────── STEP 14 期貨與借券 ─────────────────────────
def futures_sbl(full, period, ticker):
    d = _display_slice(full, period)
    start, end = d.index.min().strftime("%Y-%m-%d"), d.index.max().strftime("%Y-%m-%d")
    notes, oi, sbl = [], None, None

    tx, err = finmind_fetch("TaiwanFuturesInstitutionalInvestors", "TX", start, end)
    if err:
        notes.append(f"外資台指期：{err}")
    else:
        who = "institutional_investors" if "institutional_investors" in tx.columns else "name"
        fx = tx[tx[who].astype(str).str.contains("外資|Foreign", regex=True)]
        if fx.empty:
            notes.append(f"外資台指期：找不到外資類別（實際為 {list(tx[who].unique())}）")
        else:
            g = fx.groupby("date")[["long_open_interest_balance_volume", "short_open_interest_balance_volume"]].sum()
            oi = (g.iloc[:, 0] - g.iloc[:, 1]).reindex(d.index)

    ss, err = finmind_fetch("TaiwanDailyShortSaleBalances", ticker, start, end)
    if err:
        notes.append(f"借券賣出餘額：{err}")
    elif "SBLShortSalesCurrentDayBalance" not in ss.columns:
        notes.append("借券賣出餘額：找不到 SBLShortSalesCurrentDayBalance 欄位")
    else:
        sbl = pd.to_numeric(ss.groupby("date")["SBLShortSalesCurrentDayBalance"].last(), errors="coerce")
        if sbl.median() > 1e6:          # 單位為「股」時換算為「張」
            sbl = sbl / 1000
        sbl = sbl.reindex(d.index)

    if oi is None and sbl is None:
        raise RuntimeError("；".join(notes))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.40, 0.30, 0.30],
                        subplot_titles=("收盤價", "外資台指期淨未平倉（口）", "借券賣出餘額（張）"))
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], line=dict(color="black", width=1.5), name="收盤價"), 1, 1)
    if oi is not None:
        fig.add_trace(go.Bar(x=d.index, y=oi, name="外資台指期淨未平倉",
                             marker_color=np.where(oi.fillna(0) >= 0, UP, DOWN)), 2, 1)
    if sbl is not None:
        fig.add_trace(go.Scatter(x=d.index, y=sbl, name="借券賣出餘額", fill="tozeroy",
                                 fillcolor="rgba(128,0,128,.2)", line=dict(color="purple", width=1.5)), 3, 1)
    rb = _base_layout(fig, f"{ticker} 進階籌碼：外資台指期淨未平倉 × 借券賣出餘額", 820, d.index)
    fig.update_xaxes(rangebreaks=rb)

    lines = []
    if oi is not None and oi.notna().any():
        o = oi.dropna()
        ch = o.iloc[-1] - o.iloc[-21] if len(o) > 21 else np.nan
        lines.append(f"- **外資台指期淨未平倉**：{o.iloc[-1]:+,.0f} 口（20 日變化 {ch:+,.0f}）")
    if sbl is not None and sbl.notna().any():
        s = sbl.dropna()
        ch = s.iloc[-1] - s.iloc[-21] if len(s) > 21 else np.nan
        lines.append(f"- **借券賣出餘額**：{s.iloc[-1]:,.0f} 張（20 日變化 {ch:+,.0f}）")
    lines += [f"- ⚠️ {n}" for n in notes]
    lines.append("- 註：外資台指期部位反映整體大盤避險，並非只針對此個股；借券賣出也可能來自套利或 ETF 避險。")
    return fig, "\n".join(lines)


# ───────────────────────── 對外入口 ─────────────────────────
def build_advanced(ticker, period):
    """回傳前端用的區塊清單：[{key, title, summary, plot, error}]。"""
    full, sym = download_history(ticker, DOWNLOAD_PERIOD.get(period, "2y"))
    tasks = [
        ("bollinger", "📐 布林通道（20／60 日・收縮與放大）", lambda: bollinger(full, period, ticker)),
        ("atr", "🌊 ATR 波動度", lambda: atr(full, period, ticker)),
        ("institutional", "🏦 三大法人籌碼", lambda: institutional(full, period, ticker)),
        ("margin", "💳 融資融券", lambda: margin(full, period, ticker)),
        ("futures_sbl", "🧲 外資台指期與借券賣出", lambda: futures_sbl(full, period, ticker)),
        ("financials", "🧾 季度營收與毛利率", lambda: financials(sym, ticker)),
    ]
    out = []
    for key, title, fn in tasks:
        item = {"key": key, "title": title, "summary": "", "plot": None, "error": None}
        try:
            if full.empty and key != "financials":
                raise RuntimeError("無法下載 K 線資料")
            fig, summary = fn()
            item["summary"], item["plot"] = summary, _fig_json(fig)
        except Exception as e:
            item["error"] = str(e)
        out.append(item)
    return out
