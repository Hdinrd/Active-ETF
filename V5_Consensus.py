import pandas as pd
import glob
import os

def analyze_institutional_consensus():
    print("🔍 啟動機構共識掃描引擎...\n")
    
    csv_files = glob.glob("*_daily_holdings.csv")
    if not csv_files:
        print("⚠️ 找不到任何 CSV 檔案，請先確認爬蟲已成功執行。")
        return

    df_list = []
    
    for file in csv_files:
        etf_code = file.split('_')[0]
        df = pd.read_csv(file)
        
        # 💡 防禦機制 1：檢查是否為包含「代號」的新版 CSV
        if '代號' not in df.columns:
            print(f"⚠️ 跳過 {file}：缺少 '代號' 欄位，請刪除此舊版檔案並重新執行爬蟲。")
            continue
            
        if not df.empty:
            latest_date = df['Date'].max()
            df_latest = df[df['Date'] == latest_date].copy()
            df_latest['ETF_Code'] = etf_code
            df_list.append(df_latest)

    if not df_list:
        print("沒有有效的數據可供分析。")
        return

    all_holdings = pd.concat(df_list, ignore_index=True)
    
    # 💡 防禦機制 2：在 GroupBy 之前，強制洗淨代號型態
    # 轉字串 -> 拔除可能被 Pandas 誤加的 '.0' -> 去除頭尾空白
    all_holdings['代號'] = all_holdings['代號'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # 核心量化邏輯：強制以「代號」進行 GroupBy
    consensus_df = all_holdings.groupby('代號').agg(
        股票名稱=('股票名稱', 'first'),
        持有家數=('ETF_Code', 'nunique'),
        # 💡 防禦機制 3：加入 .unique() 確保同一家 ETF 不會被重複紀錄
        持有ETF名單=('ETF_Code', lambda x: ', '.join(x.unique())), 
        總權重=('權重', 'sum'),
        平均權重=('權重', 'mean')
    ).reset_index()

    # 排序邏輯：優先看共識度，再看資金量
    consensus_df = consensus_df.sort_values(by=['持有家數', '總權重'], ascending=[False, False])
    
    # ================= 終端機報告輸出 =================
    
    high_consensus = consensus_df[consensus_df['持有家數'] >= 4]
    print("🔥 [極度共識區] 至少被 4 家以上主動 ETF 持有:")
    if not high_consensus.empty:
        print(high_consensus[['代號', '股票名稱', '持有家數', '總權重', '持有ETF名單']].to_string(index=False))
    else:
        print("目前無高度共識標的。")
        
    print("\n" + "="*60 + "\n")
    
    mid_consensus = consensus_df[consensus_df['持有家數'] == 3]
    print("🌱 [潛在發酵區] 剛被 3 家 ETF 同時鎖定的標的:")
    if not mid_consensus.empty:
        print(mid_consensus[['代號', '股票名稱', '總權重', '持有ETF名單']].head(10).to_string(index=False))
    else:
        print("目前無潛在發酵標的。")
        
    output_file = "Market_Consensus_Latest.csv"
    consensus_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整市場共識矩陣已匯出至 {output_file}")

if __name__ == "__main__":
    analyze_institutional_consensus()