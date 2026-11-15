# -*- coding: utf-8 -*-
"""
stock.py - 模組化台股 AI 綜合量化量能觀測系統
包含數據下載、清洗、指標計算、Plotly 技術圖表繪製、Gradio Blocks 部署入口
"""

import asyncio
import asyncio.runners
import inspect
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import gradio as gr

# 確保 Gradio 伺服器啟動時與 asyncio 協程無衝突
asyncio.run = asyncio.runners.run
try:
    import uvicorn.server, uvicorn._compat
    for _m in (uvicorn.server, uvicorn._compat):
        if hasattr(_m, 'asyncio_run'):
            _m.asyncio_run = asyncio.runners.run
except ImportError:
    pass

_SYMBOL_CACHE = {}
DISPLAY_DAYS = {'1mo': 31, '3mo': 92, '6mo': 183, '1y': 365, '2y': 730}

class StockAnalyzer:
    """
    台股數據清洗與量化特徵工程核心類別
    """
    def __init__(self, ticker):
        self.ticker = str(ticker).strip().upper()
        self.df = None
        self.features = []

    @staticmethod
    def clean_data(raw):
        """
        清洗 yfinance 多層欄位，欄名轉小寫並處理還原收盤價
        """
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(subset=['open', 'high', 'low', 'close'])

    def download_data(self, period='3y'):
        """
        自動適配上市(.TW)與上櫃(.TWO)進行數據下載
        """
        if self.ticker.isdigit():
            cands = [_SYMBOL_CACHE[self.ticker]] if self.ticker in _SYMBOL_CACHE else [f'{self.ticker}.TW', f'{self.ticker}.TWO']
        else:
            cands = [self.ticker]
        for sym in cands:
            raw = yf.download(sym, period=period, auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                df_cleaned = self.clean_data(raw)
                if len(df_cleaned):
                    if self.ticker.isdigit():
                        _SYMBOL_CACHE[self.ticker] = sym
                    self.df = df_cleaned
                    return sym, self.df
        raise ValueError(f'無法取得 {self.ticker} 的歷史資料，請確認代號。')

    def calculate_indicators(self):
        """
        計算 MA、乖離率、RSI、MACD、布林通道與 ATR 波動率指標
        """
        if self.df is None:
            raise ValueError("請先執行 download_data()。")
        df = self.df.copy()
        c = df['close']
        # 移動平均線
        for n in [5, 10, 20, 60, 120, 240]:
            df[f'ma{n}'] = c.rolling(n).mean()
        
        # 20日乖離與報酬率
        df['bias_20'] = (c - df['ma20']) / df['ma20'] * 100
        df['ret_1d'] = c.pct_change(1) * 100
        df['ret_5d'] = c.pct_change(5) * 100

        # RSI (Wilder 平滑法)
        delta = c.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi14'] = 100 - 100 / (1 + avg_gain / avg_loss)

        # MACD
        df['macd_dif'] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_dif'] - df['macd_dea']

        # 布林通道
        bb_std = c.rolling(20).std()
        df['bb_up'] = df['ma20'] + 2 * bb_std
        df['bb_down'] = df['ma20'] - 2 * bb_std
        df['bb_width'] = (df['bb_up'] - df['bb_down']) / df['ma20'] * 100

        # ATR 14 波動率
        prev_c = c.shift()
        tr = pd.concat([df['high'] - df['low'],
                        (df['high'] - prev_c).abs(),
                        (df['low'] - prev_c).abs()], axis=1).max(axis=1)
        df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        self.df = df
        return df

def build_chart(df, sym):
    """
    手動雙軸排版，繪製 K線、RSI、MACD 與成交量的四層技術觀測圖表
    """
    missing = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
    red, green = '#E53935', '#2E7D32'
    fig = go.Figure()

    # 第 1 層：K 線 + 均線 + 布林通道
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'],
                                 close=df['close'], name='K線', yaxis='y',
                                 increasing_line_color=red, decreasing_line_color=green))
    for col, name, color, width, dash in [('ma5', 'MA5', 'orange', 1, None),
                                          ('ma20', 'MA20 月線', '#1565C0', 1.5, None),
                                          ('ma60', 'MA60 季線', '#8E24AA', 1.5, None),
                                          ('bb_up', '布林上軌', 'gray', 1, 'dash'),
                                          ('bb_down', '布林下軌', 'gray', 1, 'dash')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, yaxis='y',
                                 line=dict(color=color, width=width, dash=dash)))

    # 第 2 層：RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi14'], name='RSI(14)', yaxis='y2',
                             line=dict(color='purple', width=1.2)))
    for lv in (30, 70):
        fig.add_shape(type='line', xref='paper', x0=0, x1=1, yref='y2', y0=lv, y1=lv,
                      line=dict(color='gray', width=1, dash='dot'))

    # 第 3 層：MACD
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD 柱', yaxis='y3',
                         marker_color=np.where(df['macd_hist'] >= 0, red, green)))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_dif'], name='DIF', yaxis='y3',
                             line=dict(color='#1565C0', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_dea'], name='DEA', yaxis='y3',
                             line=dict(color='orange', width=1)))

    # 第 4 層：成交量
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量', yaxis='y4',
                         marker_color=np.where(df['close'] >= df['open'], red, green)))

    fig.update_layout( 
        title=f'📈 {sym} 綜合技術量能觀測盤（資料日期 {df.index[-1]:%Y/%m/%d}）',
        height=900, template='plotly_white', hovermode='x unified', bargap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=90, b=60),
        xaxis=dict(anchor='y4', rangeslider=dict(visible=False), tickformat='%Y/%m/%d', 
                   rangebreaks=[dict(values=missing.strftime('%Y-%m-%d').tolist())]),
        yaxis=dict(title='價格', domain=[0.52, 1.00]),
        yaxis2=dict(title='RSI', domain=[0.37, 0.49], range=[0, 100]),
        yaxis3=dict(title='MACD', domain=[0.20, 0.34]),
        yaxis4=dict(title='成交量', domain=[0.00, 0.17]),
    )
    return fig

# 核心分析入口與診斷評分
def analyze_stock(ticker, period):
    try:
        analyzer = StockAnalyzer(ticker)
        sym, raw = analyzer.download_data('3y')
        full = analyzer.calculate_indicators()
        df = full[full.index >= full.index[-1] - pd.Timedelta(days=DISPLAY_DAYS.get(period, 183))]
        if len(df) < 2:
            raise ValueError('資料筆數不足。')

        latest, prev = df.iloc[-1], df.iloc[-2]
        chg = (latest['close'] / prev['close'] - 1) * 100
        atr_stop = latest['close'] - 2 * latest['atr14']
        c, m20, m60 = latest['close'], latest['ma20'], latest['ma60']
        
        # 趨勢與技術總評判定
        if pd.isna(m60):
            trend = '資料不足'
        elif c > m20 > m60:
            trend = '多頭排列'
        elif c < m20 < m60:
            trend = '空頭排列'
        else:
            trend = '均線糾結'

        # 數值映射與燈號評估
        score = 50
        if trend == '多頭排列': score += 20
        elif trend == '空頭排列': score -= 20
        
        if latest['rsi14'] > 65: score += 15 if latest['rsi14'] < 80 else 5
        elif latest['rsi14'] < 35: score -= 15 if latest['rsi14'] > 20 else -5
        else: score += (latest['rsi14'] - 50) * 0.5
        
        if latest['macd_hist'] > 0: score += 10
        else: score -= 10
        
        score = min(max(score, 0), 100)
        
        if score >= 75:
            rating, strategy = "🟢 強勢多頭控盤", "建議順勢做多、守月線防守，波動拉回不破均線視為買點。"
        elif score >= 55:
            rating, strategy = "🟡 中性偏多震盪", "價格位處均線上方，適合在支撐線附近分批佈局。"
        elif score >= 40:
            rating, strategy = "⚪ 區間震盪整理", "多空交戰均線糾結，建議以高拋低吸的區間策略操作。"
        else:
            rating, strategy = "🔴 空頭修正弱勢", "價格反壓沉重，建議保留現金成數或減碼防守，不宜盲目摸底。"

        table_data = pd.DataFrame({
            '量化核心指標': ['最新收盤價', '單日漲跌幅', '5日累積報酬', '20日乖離率', '均線排列',
                           'RSI(14)', 'MACD 柱', '布林帶寬', 'ATR(14)', '量化綜合評分'],
            '數值': [f"{c:.2f}", f"{chg:+.2f}%", f"{latest['ret_5d']:+.2f}%",
                    f"{latest['bias_20']:+.2f}%", f"MA20 {m20:.1f}／MA60 {m60:.1f}",
                    f"{latest['rsi14']:.1f}", f"{latest['macd_hist']:.2f}",
                    f"{latest['bb_width']:.2f}%", f"{latest['atr14']:.2f}", f"{score:.1f} / 100"],
            '狀態': [f"資料日期 {df.index[-1]:%Y/%m/%d}",
                    '上漲' if chg > 0 else '下跌',
                    '累積上漲' if latest['ret_5d'] > 0 else '累積下跌',
                    '正乖離偏高' if latest['bias_20'] > 5 else ('負乖離偏大' if latest['bias_20'] < -5 else '均線常態軌道'),
                    trend, rating, '多方動能' if latest['macd_hist'] > 0 else '空方動能',
                    '波動擴張' if latest['bb_width'] > 10 else '帶寬收縮',
                    f'參考防守價 {atr_stop:.2f}', rating],
        })

        md_text = f"""### 🔍 {sym} 量化分析診斷
- **綜合技術評定**：**{rating}**（量化分數：`{score:.1f}` 分）
- **趨勢狀態**：均線結構呈現 **{trend}**；20 日乖離率為 **{latest['bias_20']:+.2f}%**。
- **風控建議**：當前 14 日 ATR 為 `{latest['atr14']:.2f}` 元，停損/防守參考價設在 **{atr_stop:.2f}** 元。
- **💡 操作指引**：{strategy}
""" 
        return build_chart(df, sym), table_data, md_text
    except Exception as e:
        empty = go.Figure().update_layout(title='資料載入或計算失敗，請更換代號重試。')
        return empty, pd.DataFrame({'錯誤': [str(e)]}), f'⚠️ **執行錯誤**：{e}'

def scan_watchlist(watchlist_str):
    """
    自選股多空雷達掃描核心
    """
    codes = [t.strip() for t in str(watchlist_str).replace('，', ',').split(',') if t.strip()]
    rows, failed = [], []
    for code in codes:
        try:
            analyzer = StockAnalyzer(code)
            sym, raw = analyzer.download_data('6mo')
            d = analyzer.calculate_indicators()
            last, prev = d.iloc[-1], d.iloc[-2]
            chg = (last['close'] / prev['close'] - 1) * 100
            score = int(chg > 0) + int(last['close'] > last['ma20']) + int(45 <= last['rsi14'] <= 65)
            rows.append({
                '代號': sym,
                '資料日期': f'{d.index[-1]:%Y/%m/%d}',
                '收盤價': round(last['close'], 2),
                '單日漲跌%': round(chg, 2),
                'MA20': round(last['ma20'], 2),
                '20日乖離%': round(last['bias_20'], 2),
                'RSI(14)': round(last['rsi14'], 1),
                '多空狀態': {3: '🟢 多方強勢', 2: '🟡 中性整理'}.get(score, '🔴 空方弱勢'),
            })
        except Exception as e:
            failed.append(f'{code}：{e}')
    msg = f'✅ 成功掃描 {len(rows)} / {len(codes)} 檔'
    if failed:
        msg += '

⚠️ 出錯代號：
' + '
'.join(f'- {f}' for f in failed)
    if not rows:
        return pd.DataFrame({'提示': ['未取得任何有效資料']}), msg
    return pd.DataFrame(rows), msg

# ── Gradio 介面配置與啟動門口 ──
def build_gradio_app():
    with gr.Blocks(title='台股 AI 綜合量化量能觀測儀表板', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 📈 台股 AI 綜合量化量能觀測儀表板 (獨立部署版)')
        with gr.Tabs():
            with gr.Tab('📊 單檔技術透視'):
                with gr.Row():
                    with gr.Column(scale=1):
                        ticker_input = gr.Textbox(value='2330', label='股票代號 (上市 2330 / 上櫃 6488.TWO)')
                        period_select = gr.Dropdown(choices=list(DISPLAY_DAYS), value='6mo', label='顯示區間')
                        analyze_btn = gr.Button('🚀 啟動綜合量化診斷', variant='primary')
                    with gr.Column(scale=3):
                        diagnostic_text = gr.Markdown('### 🕒 請點擊左側啟動診斷...')
                with gr.Row():
                    with gr.Column(scale=3):
                        plotly_chart = gr.Plot(label='技術指標全景圖')
                    with gr.Column(scale=1):
                        indicator_table = gr.Dataframe(interactive=False, wrap=True)
                analyze_btn.click(analyze_stock, inputs=[ticker_input, period_select], outputs=[plotly_chart, indicator_table, diagnostic_text])

            with gr.Tab('🎯 智慧自選股雷達'):
                watchlist_input = gr.Textbox(value='2330, 2317, 2454, 2303, 2308', label='自選股清單（逗號分隔）')
                scan_btn = gr.Button('⚡ 執行批次雷達掃描', variant='secondary')
                radar_msg = gr.Markdown()
                radar_table = gr.Dataframe(interactive=False, wrap=True)
                scan_btn.click(scan_watchlist, inputs=[watchlist_input], outputs=[radar_table, radar_msg])
    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch()