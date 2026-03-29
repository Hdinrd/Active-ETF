import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from FinMind.data import DataLoader
from scipy.stats import norm

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Dashboard Pro")

# Sidebar Navigation (這是做分頁最簡單直白的方式)
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Correlation Lab", "ETF Consensus"])

# Global Sidebar Parameters
st.sidebar.header("Global Parameters")
# 這些參數主要給 Dashboard 用，但也可以設為全域
ticker_input = st.sidebar.text_input("Main Ticker (e.g., 2330)", "2330") 
lookback = st.sidebar.slider("Lookback Days", 200, 1000, 365)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_market_data(stock_id, days):   
    # Source A: yfinance
    yf_ticker = f"{stock_id}.TW"
    try:
        df_price = yf.download(yf_ticker, period=f"{days}d", progress=False)
        # Fix for yfinance multi-level columns
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = df_price.columns.droplevel('Ticker')
        df_price.reset_index(inplace=True)
    except:
        return None, None

    # Source B: FinMind (Chips)
    try:
        api = DataLoader()
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        df_chips = api.taiwan_stock_institutional_investors(
            stock_id=stock_id, 
            start_date=start_date
        )
        if not df_chips.empty:
            df_chips['date'] = pd.to_datetime(df_chips['date'])
            df_chips = df_chips.groupby(['date', 'name'])['buy'].sum() - df_chips.groupby(['date', 'name'])['sell'].sum()
            df_chips = df_chips.unstack(level=1).fillna(0)
            df_chips['Total_Net_Buy'] = df_chips.sum(axis=1)
            df_chips.reset_index(inplace=True)
        else:
            df_chips = None
    except Exception as e:
        # st.warning(f"FinMind API Warning: {e}") # Silent fail is cleaner sometimes
        df_chips = None
        
    return df_price, df_chips

# --- PAGE 1: MAIN DASHBOARD (原本的功能) ---
if page == "Dashboard":
    st.title("🚀 Quant Dashboard: Monte Carlo & Chips") 
    st.markdown("Feature: Signal Processing (MACD) + External Info (Chips) + Channel Modeling (Monte Carlo)")

    # Local Sidebar params for this page
    mc_sims = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 500)
    forecast_days = st.sidebar.slider("Forecast Horizon", 10, 90, 30)

    # Load Data
    df_price, df_chips = load_market_data(ticker_input, lookback)

    if df_price is None or df_price.empty:
        st.error("Error loading data. Check ticker.")
        st.stop()

    # Data Merge
    if df_chips is not None:
        df = pd.merge(df_price, df_chips, left_on='Date', right_on='date', how='left')
        df['Total_Net_Buy'] = df['Total_Net_Buy'].fillna(0)
    else:
        df = df_price
        df['Total_Net_Buy'] = 0

    # Signal Processing
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()

    # Monte Carlo Logic
    def run_monte_carlo(data, days_forecast, num_simulations):
        log_returns = np.log(1 + data['Close'].pct_change())
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        
        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days_forecast, num_simulations)))
        
        last_price = data['Close'].iloc[-1]
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = last_price
        
        for t in range(1, days_forecast):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
        return price_paths

    simulation_paths = run_monte_carlo(df, forecast_days, mc_sims)
    final_prices = simulation_paths[-1]

    # Metrics
    current_price = df['Close'].iloc[-1]
    expected_price = np.mean(final_prices)
    upside_prob = np.sum(final_prices > current_price) / len(final_prices)
    var_95 = np.percentile(final_prices, 5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{current_price:.1f}")
    col2.metric("MC Expected Price", f"{expected_price:.1f}", delta=f"{expected_price-current_price:.1f}")
    col3.metric("Upside Probability", f"{upside_prob*100:.1f}%")
    col4.metric("95% VaR", f"{var_95:.1f}", delta_color="inverse")

    # Plots
    tab1, tab2, tab3 = st.tabs(["Technical & Chips", "Monte Carlo Paths", "Probability Dist."])
    
    with tab1:
        st.subheader("Price Action + Institutional Chips")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'))
        chip_colors = ['red' if x > 0 else 'green' for x in df['Total_Net_Buy']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['Total_Net_Buy'], name='Net Buy (Chips)',
                             marker_color=chip_colors, yaxis='y2', opacity=0.3))
        fig.update_layout(yaxis2=dict(title="Volume/Chips", overlaying='y', side='right'), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Monte Carlo: {mc_sims} Simulations")
        fig_mc = go.Figure()
        for i in range(min(50, mc_sims)):
            fig_mc.add_trace(go.Scatter(y=simulation_paths[:, i], mode='lines', 
                                        line=dict(width=1, color='rgba(0,100,255,0.2)'), showlegend=False))
        fig_mc.add_trace(go.Scatter(y=np.mean(simulation_paths, axis=1), mode='lines', 
                                    name='Mean Path', line=dict(color='orange', width=3)))
        st.plotly_chart(fig_mc, use_container_width=True)

    with tab3:
        fig_hist = px.histogram(final_prices, nbins=50, title="Distribution of Final Prices")
        fig_hist.add_vline(x=current_price, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

# --- PAGE 2: CORRELATION LAB (新增的功能) ---
# --- PAGE 2: CORRELATION LAB (Fixed Layout) ---
elif page == "Correlation Lab":
    st.title("🧪 Correlation Lab (Cross-Market)")
    st.markdown("""
    Compare **TW Stocks** vs **US Stocks**.
    *Tickers with digits only -> Auto-add .TW suffix.*
    """)
    
    # 輸入區塊放進一個 expander 或獨立 column 比較整潔
    with st.container():
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            # 預設範例：台積電生態系 + 競品
            default_tickers = "2330, 2454, 2317, TSM, NVDA" 
            tickers_string = st.text_input("Enter Tickers (comma separated)", default_tickers)
        with col_btn:
            st.write("") # Spacer
            st.write("") # Spacer
            run_btn = st.button("Calculate Cross-Market Data", type="primary")
    
    if run_btn:
        ticker_list = [t.strip() for t in tickers_string.split(",")]
        ticker_list_formatted = [f"{t}.TW" if t.isdigit() else t for t in ticker_list]
        
        st.info(f"Fetching: {ticker_list_formatted} (Auto-aligned time zones)")
        
        try:
            data = yf.download(ticker_list_formatted, period=f"{lookback}d")['Close']
            
            # --- 修正開始 (Fix Start) ---
            # 1. 處理欄位名稱
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            data.columns = [c.replace('.TW', '') for c in data.columns]

            # 🔥🔥🔥 關鍵修正：強制轉成字串，避免 Plotly 把它當數字畫座標軸 🔥🔥🔥
            data.columns = data.columns.astype(str)
            
            # 2. 補值 & 算相關性
            df_filled = data.ffill()
            df_corr = df_filled.pct_change().dropna()
            df_plot = df_filled.dropna()
            # --- 修正結束 (Fix End) ---

            # ... (後面畫圖程式碼不用動)

            # --- Analysis 1: Correlation Matrix ---
            st.subheader("1. Return Correlation Matrix")
            
            c1, c2, c3 = st.columns([1, 2, 1])
            
            # ... (上面程式碼不變)
            
            with c2:
                corr_matrix = df_corr.corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f",
                    aspect="equal",
                    color_continuous_scale="RdBu_r", 
                    zmin=-1, zmax=1,
                    title="Correlation Heatmap"
                )
                
                # 🔥🔥🔥 加入這兩行，強制鎖定為「類別軸」 🔥🔥🔥
                # 這樣不管你的代號長得像不像數字，它都會乖乖顯示名字
                fig_corr.update_xaxes(side="bottom", type="category")
                fig_corr.update_yaxes(side="left", type="category")
                
                fig_corr.update_layout(height=500) 
                st.plotly_chart(fig_corr, use_container_width=True)
                
            # ... (下面程式碼不變)
            
            # (後面程式碼不變...)
            
            # --- Layout Fix 2: Price Trend (Full Width but controlled) ---
            st.subheader("2. Cumulative Return Comparison")
            
            normalized_data = (df_plot / df_plot.iloc[0]) * 100
            
            fig_norm = go.Figure()
            for col in normalized_data.columns:
                fig_norm.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], name=col))
                
            fig_norm.update_layout(
                yaxis_title="Normalized Price (Base=100)",
                hovermode="x unified",
                height=600, # 固定高度，視覺更穩定
                legend=dict(
                    orientation="h", # 圖例放上面，節省水平空間
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_norm, use_container_width=True)
            
            # --- Analysis 3: Optional Ratio (Side by Side) ---
            if "2330" in data.columns and "TSM" in data.columns:
                st.subheader("3. Arbitrage Analysis")
                c_a, c_b = st.columns(2)
                
                with c_a:
                     ratio = df_plot['TSM'] / df_plot['2330']
                     fig_ratio = px.line(ratio, title="TSM / 2330 Price Ratio")
                     fig_ratio.update_layout(height=400)
                     st.plotly_chart(fig_ratio, use_container_width=True)
                
                with c_b:
                     # 這裡可以放個簡單的統計或是 Scatter
                     fig_scatter = px.scatter(df_corr, x='2330', y='TSM', title="Daily Returns Scatter", trendline="ols")
                     fig_scatter.update_layout(height=400)
                     st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}") 
# --- PAGE 3: ETF CONSENSUS HEATMAP (新增的熱力圖分頁) ---
elif page == "ETF Consensus":
    st.title("🔥 主動型 ETF 機構共識熱力圖")
    st.markdown("透過橫截面 (Cross-sectional) 掃描 7 大投信，抓出 Buy-side 資金最擁擠或剛發酵的板塊。")

    import glob
    
    # 讀取本地端所有的 CSV 矩陣
    csv_files = glob.glob("*_daily_holdings.csv")
    
    if not csv_files:
        st.warning("⚠️ 找不到本地端的 CSV 資料，請先執行爬蟲擷取引擎 (V4_Batch.py)。")
    else:
        df_list = []
        for file in csv_files:
            etf_code = file.split('_')[0]
            df = pd.read_csv(file)
            
            # 確保是新版帶有 '代號' 欄位的資料
            if '代號' in df.columns and not df.empty:
                latest_date = df['Date'].max()
                df_latest = df[df['Date'] == latest_date].copy()
                df_latest['ETF_Code'] = etf_code
                df_list.append(df_latest)
                
        if df_list:
            all_holdings = pd.concat(df_list, ignore_index=True)
            # 資料清洗：對齊代號型態
            all_holdings['代號'] = all_holdings['代號'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
            # 建立 Pivot Table (X軸: ETF, Y軸: 股票, Value: 權重)
            pivot_df = all_holdings.pivot_table(
                index='股票名稱', 
                columns='ETF_Code', 
                values='權重', 
                aggfunc='sum'
            ).fillna(0)
            
            # 計算共識度與資金總熱度
            pivot_df['持有家數'] = (pivot_df > 0).sum(axis=1)
            pivot_df['總權重'] = pivot_df.drop(columns=['持有家數'], errors='ignore').sum(axis=1)
            
            # UI：側邊欄濾網
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Heatmap 濾網")
            min_holders = st.sidebar.slider("最小持有家數 (共識度)", min_value=1, max_value=7, value=2)
            top_n = st.sidebar.number_input("顯示總權重 Top N", min_value=10, max_value=100, value=40)
            
            # 套用濾網
            filtered_df = pivot_df[pivot_df['持有家數'] >= min_holders]
            filtered_df = filtered_df.sort_values(by='總權重', ascending=False).head(top_n)
            
            # 準備畫圖 (拔除輔助計算欄位)
            heatmap_data = filtered_df.drop(columns=['持有家數', '總權重'])
            
            # 繪製 Plotly 熱力圖
            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="主動型 ETF", y="股票名稱", color="權重(%)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="Reds",
                aspect="auto",
                text_auto=".2f" # 直接在格子上顯示權重數字
            )
            # 鎖定左側 Y 軸為類別，避免被當成數字排序
            fig_heat.update_yaxes(type="category")
            fig_heat.update_layout(height=800) 
            
            st.plotly_chart(fig_heat, use_container_width=True)
            
            with st.expander("📊 查看底層共識數據矩陣"):
                st.dataframe(filtered_df.style.background_gradient(cmap='Reds', axis=None))