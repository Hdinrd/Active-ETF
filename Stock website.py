import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from FinMind.data import DataLoader
from scipy.stats import norm
import glob
import os

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Dashboard Pro", page_icon="🚀")

# Sidebar Navigation
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Correlation Lab", "ETF Consensus", "Alpha Terminal"])

# Global Sidebar Parameters
st.sidebar.header("Global Parameters")
ticker_input = st.sidebar.text_input("Main Ticker (e.g., 2330)", "2330") 
lookback = st.sidebar.slider("Lookback Days", 200, 1000, 365)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_market_data(stock_id, days):   
    # Source A: yfinance
    yf_ticker = f"{stock_id}.TW"
    try:
        df_price = yf.download(yf_ticker, period=f"{days}d", progress=False)
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
        df_chips = None
        
    return df_price, df_chips

@st.cache_data(ttl=3600)
def load_consensus_data():
    file_path = "Market_Consensus_Latest.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'代號': str})
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_confluence_signals():
    csv_files = glob.glob("*_daily_holdings.csv")
    if not csv_files: return pd.DataFrame()
    
    all_data = []
    for file in csv_files:
        etf_code = file.split('_')[0]
        df = pd.read_csv(file, dtype={'代號': str})
        df['ETF'] = etf_code
        all_data.append(df)
        
    df_all = pd.concat(all_data, ignore_index=True)
    dates = sorted(df_all['Date'].unique())
    if len(dates) < 2: return pd.DataFrame()
    
    t_0, t_1 = dates[-1], dates[-2]
    df_t0 = df_all[df_all['Date'] == t_0]
    df_t1 = df_all[df_all['Date'] == t_1]
    name_map = df_t0.groupby('代號')['股票名稱'].first().to_dict()

    t1_holders = df_t1.groupby('代號')['ETF'].apply(list).reset_index(name='T1_ETFs')
    t1_holders['T1_Count'] = t1_holders['T1_ETFs'].apply(len)
    
    t0_holders = df_t0.groupby('代號')['ETF'].apply(list).reset_index(name='T0_ETFs')
    t0_holders['T0_Count'] = t0_holders['T0_ETFs'].apply(len)

    merged = pd.merge(t1_holders, t0_holders, on='代號', how='outer').fillna({'T1_Count': 0, 'T0_Count': 0})
    merged['T1_ETFs'] = merged['T1_ETFs'].apply(lambda d: d if isinstance(d, list) else [])
    merged['T0_ETFs'] = merged['T0_ETFs'].apply(lambda d: d if isinstance(d, list) else [])

    # V9 核心濾網
    confluence_df = merged[(merged['T1_Count'] <= 2) & (merged['T0_Count'] > merged['T1_Count'])].copy()
    
    if confluence_df.empty: return pd.DataFrame()
    
    confluence_df['股票名稱'] = confluence_df['代號'].map(name_map)
    confluence_df['點火主力'] = confluence_df.apply(lambda row: ", ".join(set(row['T0_ETFs']) - set(row['T1_ETFs'])), axis=1)
    
    return confluence_df[['代號', '股票名稱', 'T1_Count', 'T0_Count', '點火主力']].sort_values(by='T0_Count', ascending=False)

# 💡 載入 V7 3D 流動性衝擊資料
@st.cache_data(ttl=3600)
def load_impact_data():
    file_path = "V7_Impact_Results.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={'代號': str})
    return pd.DataFrame()


# --- PAGE 1: MAIN DASHBOARD ---
if page == "Dashboard":
    st.title("🚀 Quant Dashboard: Monte Carlo & Chips") 
    st.markdown("Feature: Signal Processing (MACD) + External Info (Chips) + Channel Modeling (Monte Carlo)")

    mc_sims = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 500)
    forecast_days = st.sidebar.slider("Forecast Horizon", 10, 90, 30)

    df_price, df_chips = load_market_data(ticker_input, lookback)

    if df_price is None or df_price.empty:
        st.error("Error loading data. Check ticker.")
        st.stop()

    if df_chips is not None:
        df = pd.merge(df_price, df_chips, left_on='Date', right_on='date', how='left')
        df['Total_Net_Buy'] = df['Total_Net_Buy'].fillna(0)
    else:
        df = df_price
        df['Total_Net_Buy'] = 0

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()

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

    current_price = df['Close'].iloc[-1]
    expected_price = np.mean(final_prices)
    upside_prob = np.sum(final_prices > current_price) / len(final_prices)
    var_95 = np.percentile(final_prices, 5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{current_price:.1f}")
    col2.metric("MC Expected Price", f"{expected_price:.1f}", delta=f"{expected_price-current_price:.1f}")
    col3.metric("Upside Probability", f"{upside_prob*100:.1f}%")
    col4.metric("95% VaR", f"{var_95:.1f}", delta_color="inverse")

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


# --- PAGE 2: CORRELATION LAB ---
elif page == "Correlation Lab":
    st.title("🧪 Correlation Lab (Cross-Market)")
    st.markdown("""
    Compare **TW Stocks** vs **US Stocks**.
    *Tickers with digits only -> Auto-add .TW suffix.*
    """)
    
    with st.container():
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            default_tickers = "2330, 2454, 2317, TSM, NVDA" 
            tickers_string = st.text_input("Enter Tickers (comma separated)", default_tickers)
        with col_btn:
            st.write("") 
            st.write("") 
            run_btn = st.button("Calculate Cross-Market Data", type="primary")
    
    if run_btn:
        ticker_list = [t.strip() for t in tickers_string.split(",")]
        ticker_list_formatted = [f"{t}.TW" if t.isdigit() else t for t in ticker_list]
        
        st.info(f"Fetching: {ticker_list_formatted} (Auto-aligned time zones)")
        
        try:
            data = yf.download(ticker_list_formatted, period=f"{lookback}d")['Close']
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            data.columns = [c.replace('.TW', '') for c in data.columns]
            data.columns = data.columns.astype(str)
            
            df_filled = data.ffill()
            df_corr = df_filled.pct_change().dropna()
            df_plot = df_filled.dropna()

            st.subheader("1. Return Correlation Matrix")
            c1, c2, c3 = st.columns([1, 2, 1])
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
                fig_corr.update_xaxes(side="bottom", type="category")
                fig_corr.update_yaxes(side="left", type="category")
                fig_corr.update_layout(height=500) 
                st.plotly_chart(fig_corr, use_container_width=True)
                
            st.subheader("2. Cumulative Return Comparison")
            normalized_data = (df_plot / df_plot.iloc[0]) * 100
            
            fig_norm = go.Figure()
            for col in normalized_data.columns:
                fig_norm.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], name=col))
                
            fig_norm.update_layout(
                yaxis_title="Normalized Price (Base=100)",
                hovermode="x unified",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_norm, use_container_width=True)
            
            if "2330" in data.columns and "TSM" in data.columns:
                st.subheader("3. Arbitrage Analysis (TSM vs 2330)")
                c_a, c_b = st.columns(2)
                
                with c_a:
                     ratio = df_plot['TSM'] / df_plot['2330']
                     fig_ratio = px.line(ratio, title="TSM / 2330 Price Ratio")
                     fig_ratio.update_layout(height=400)
                     st.plotly_chart(fig_ratio, use_container_width=True)
                
                with c_b:
                     fig_scatter = px.scatter(df_corr, x='2330', y='TSM', title="Daily Returns Scatter", trendline="ols")
                     fig_scatter.update_layout(height=400)
                     st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}") 

# --- PAGE 3: ETF CONSENSUS HEATMAP ---
elif page == "ETF Consensus":
    st.title("🔥 主動型 ETF 機構共識熱力圖")
    st.markdown("透過橫截面 (Cross-sectional) 掃描 7 大投信，抓出 Buy-side 資金最擁擠或剛發酵的板塊。")
    
    csv_files = glob.glob("*_daily_holdings.csv")
    
    if not csv_files:
        st.warning("⚠️ 找不到本地端的 CSV 資料，請先執行爬蟲擷取引擎。")
    else:
        df_list = []
        for file in csv_files:
            etf_code = file.split('_')[0]
            df = pd.read_csv(file)
            
            if '代號' in df.columns and not df.empty:
                latest_date = df['Date'].max()
                df_latest = df[df['Date'] == latest_date].copy()
                df_latest['ETF_Code'] = etf_code
                df_list.append(df_latest)
                
        if df_list:
            all_holdings = pd.concat(df_list, ignore_index=True)
            all_holdings['代號'] = all_holdings['代號'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            ticker_to_name = all_holdings.groupby('代號')['股票名稱'].first()
            
            pivot_df = all_holdings.pivot_table(
                index='代號', 
                columns='ETF_Code', 
                values='權重', 
                aggfunc='sum'
            ).fillna(0)
            
            pivot_df.index = pivot_df.index.astype(str) + " " + pivot_df.index.map(ticker_to_name)
            
            pivot_df['持有家數'] = (pivot_df > 0).sum(axis=1)
            pivot_df['總權重'] = pivot_df.drop(columns=['持有家數'], errors='ignore').sum(axis=1)
            
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Heatmap 濾網")
            min_holders = st.sidebar.slider("最小持有家數 (共識度)", min_value=1, max_value=7, value=2)
            top_n = st.sidebar.number_input("顯示總權重 Top N", min_value=10, max_value=100, value=40)
            
            filtered_df = pivot_df[pivot_df['持有家數'] >= min_holders]
            filtered_df = filtered_df.sort_values(by='總權重', ascending=False).head(top_n)
            
            heatmap_data = filtered_df.drop(columns=['持有家數', '總權重'])
            
            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="主動型 ETF", y="標的", color="權重(%)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="Reds",
                aspect="auto",
                text_auto=".2f"
            )
            fig_heat.update_yaxes(type="category")
            fig_heat.update_layout(height=800) 
            
            st.plotly_chart(fig_heat, use_container_width=True)
            
            with st.expander("📊 查看底層共識數據矩陣"):
                st.dataframe(filtered_df.style.background_gradient(cmap='Reds', axis=None))

# --- PAGE 4: ALPHA TERMINAL (V12 Integration) ---
elif page == "Alpha Terminal":
    st.title("🚀 Alpha 戰情室：機構微結構透視終端")
    st.markdown("---")
    
    df_consensus = load_consensus_data()
    df_confluence = load_confluence_signals()
    df_impact = load_impact_data()  # 💡 載入 V7 數據

    # 💡 新增第三個 Tab
    tab1, tab2, tab3 = st.tabs(["🔥 橫截面共識熱力圖", "🎯 完美擊球點掃描", "🌊 3D 流動性衝擊矩陣"])

    with tab1:
        st.subheader("📊 機構持股板塊分佈")
        if not df_consensus.empty:
            plot_df = df_consensus[df_consensus['持有家數'] >= 3].copy()
            plot_df['標籤'] = plot_df['股票名稱'] + "<br>(" + plot_df['持有家數'].astype(str) + "家)"
            
            fig = px.treemap(
                plot_df, 
                path=[px.Constant("全市場高共識標的"), '持有家數', '標籤'], 
                values='總權重',
                color='持有家數',
                color_continuous_scale='Turbo',
                title="區塊大小 = 總權重 | 顏色深淺 = 持有家數"
            )
            fig.update_traces(textinfo="label+value")
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("尚無共識資料，請先執行 Alpha 引擎！")

    with tab2:
        st.subheader("🎯 今日籌碼共振突破名單 (V9)")
        if not df_confluence.empty:
            st.dataframe(
                df_confluence.rename(columns={'T1_Count': '昨日持有家數', 'T0_Count': '今日持有家數'}),
                use_container_width=True,
                hide_index=True
            )
            st.info("💡 Quant 提示：請配合晨報中的【V7 市場衝擊成本】與【外資環境標籤】，剃除參與率 > 10% 或是外資大量倒貨的標的。")
        else:
            st.success("今日無符合條件的初升段共振標的。")

    with tab3:
        st.subheader("🌊 投信資金微結構解析 (V7 3D 衝擊矩陣)")
        if not df_impact.empty:
            # 讓 DataFrame 在網頁上看起來更專業
            st.dataframe(
                df_impact,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "參與率(%)": st.column_config.ProgressColumn(
                        "參與率(%)", help="佔單日總成交量比例", format="%.2f%%", min_value=0, max_value=25
                    ),
                    "漲跌幅(%)": st.column_config.NumberColumn(
                        "漲跌幅(%)", format="%.2f%%"
                    ),
                    "最新總權重(%)": st.column_config.NumberColumn(
                        "最新總權重(%)", format="%.2f%%"
                    )
                }
            )
            st.info("💡 尋寶指南：優先鎖定【🌱 完美初升段】與【🛡️ 穩健底倉】，避開【🚨 異常擁擠】與【💀 提款機陷阱】。")
        else:
            st.warning("尚無 V7 衝擊分析資料，請先執行本地端 Alpha 引擎產生數據。")