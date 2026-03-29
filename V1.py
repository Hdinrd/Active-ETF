import pandas as pd
import os
import io
import time
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# 💡 目標矩陣：台灣目前主動式 ETF 前段班代碼 (台股聚焦)
# 包含：野村(00980A)、統一(00981A)、群益(00982A, 00992A)、復華(00991A)、元大(00990A)、安聯(00993A)
TARGET_ETFS = ["00980A", "00981A", "00982A", "00991A", "00992A", "00990A", "00993A"]

def fetch_batch_holdings(etf_list: list):
    """
    共用單一 Browser Instance，批次高速擷取多家 ETF 全量持股
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    print(f"[{today_str}] 🚀 啟動 7 大主動 ETF 矩陣擷取引擎...\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context() # 建立乾淨的瀏覽上下文
        
        for etf in etf_list:
            print(f"🔄 正在處理 {etf}...")
            url = f"https://www.pocket.tw/etf/tw/{etf}/fundholding"
            page = context.new_page()
            
            try:
                page.goto(url)
                page.wait_for_load_state('networkidle')
                # 給一點彈性時間尋找表格，找不到就跳過 (代表該 ETF 可能無資料或代碼錯誤)
                page.wait_for_selector("table", timeout=5000)
                
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                tables = pd.read_html(io.StringIO(str(soup)))
                
                # 解析與清洗
                df_target = pd.DataFrame()
                for df in tables:
                    if '名稱' in df.columns and '權重' in df.columns:
                        df = df.rename(columns={'名稱': '股票名稱'})
                        df['權重'] = df['權重'].astype(str).str.replace('%', '', regex=False)
                        df['權重'] = pd.to_numeric(df['權重'], errors='coerce')
                        df = df.dropna(subset=['權重'])
                        df = df[df['權重'] > 0]
                        df['Date'] = today_str
                        # 擷取 Quant 需要的欄位
                        df_target = df[['代號', '股票名稱', '權重', '持有數', 'Date']]
                        break # 找到正確表格就跳出內部迴圈
                
                # 儲存邏輯 (獨立存成各自的 CSV)
                if not df_target.empty:
                    csv_file = f"{etf}_daily_holdings.csv"
                    # 清理當日舊資料防呆
                    if os.path.exists(csv_file):
                        old_df = pd.read_csv(csv_file)
                        old_df = old_df[old_df['Date'] != today_str]
                        old_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                        
                    df_target.to_csv(csv_file, mode='a', index=False, encoding='utf-8-sig', header=not os.path.exists(csv_file))
                    print(f"   ✅ {etf} 擷取完成，共 {len(df_target)} 檔標的")
                    
                    # 呼叫 Delta 運算 (若歷史資料 >= 2 天)
                    calculate_flow_delta(csv_file, etf)
                else:
                    print(f"   ⚠️ {etf} 解析失敗，未找到有效表格。")
                    
            except Exception as e:
                print(f"   ❌ {etf} 發生錯誤: 網頁載入逾時或無此資料。")
            
            finally:
                page.close()
                # 💡 Risk Control：隨機休眠 2~3 秒，避免密集 Request 被 Server Ban IP
                time.sleep(2)
                
        browser.close()
        print("\n🎉 全矩陣擷取任務結束。")

def calculate_flow_delta(csv_path: str, etf_name: str):
    """
    計算籌碼變動，加上 ETF 名稱標籤方便辨識
    """
    if not os.path.exists(csv_path): return
    df_all = pd.read_csv(csv_path)
    dates = sorted(df_all['Date'].unique())
    if len(dates) < 2: return
        
    t_0_date, t_1_date = dates[-1], dates[-2]
    df_t0 = df_all[df_all['Date'] == t_0_date].set_index('股票名稱')
    df_t1 = df_all[df_all['Date'] == t_1_date].set_index('股票名稱')
    
    delta_matrix = df_t0[['權重']].join(df_t1[['權重']], lsuffix='_今日', rsuffix='_昨日', how='outer').fillna(0)
    delta_matrix['權重變化(%)'] = delta_matrix['權重_今日'] - delta_matrix['權重_昨日']
    
    def categorize_action(row):
        if row['權重_昨日'] == 0 and row['權重_今日'] > 0: return "🆕 建倉"
        elif row['權重_昨日'] > 0 and row['權重_今日'] == 0: return "🗑️ 清倉"
        elif row['權重變化(%)'] > 0: return "🔼 加碼"
        elif row['權重變化(%)'] < 0: return "🔽 減碼"
        else: return "⏸️ 不變"
            
    delta_matrix['動向'] = delta_matrix.apply(categorize_action, axis=1)
    delta_matrix = delta_matrix.reindex(delta_matrix['權重變化(%)'].abs().sort_values(ascending=False).index)
    
    active_changes = delta_matrix[delta_matrix['權重變化(%)'] != 0]
    if not active_changes.empty:
        print(f"   📊 [{etf_name} 動能] 前 3 大異動: " + 
              ", ".join([f"{idx}({row['動向']}{row['權重變化(%)']:.2f}%)" for idx, row in active_changes.head(3).iterrows()]))

if __name__ == "__main__":
    fetch_batch_holdings(TARGET_ETFS)