import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Root Cause Mini Tool", layout="wide")

########################################################
# 0. 公用邏輯：清洗、判斷欄位型態、彙總
########################################################

ERROR_TOKENS = [
    "", " ", "  ", "\t",
    "#DIV/0!", "#DIV/0", "#VALUE!", "#REF!", "#NAME?", "#N/A",
    "#NUM!", "#NULL!", "ERROR", "Error", "error"
]

def normalize_cell_value(v):
    """
    規則：
    - 空值/空白/錯誤(#DIV/0! 等)/'Error' -> 0
    - 其他維持原狀（不管是中文、英文、料號、客戶名都保留）
    """
    if pd.isna(v):
        return 0
    if isinstance(v, str):
        vs = v.strip()
        # 完整比對錯誤/空白字串
        for bad in ERROR_TOKENS:
            if vs == bad:
                return 0
        # fallback：用大寫防一些變形 (#div/0!   等)
        if vs.upper() in [t.upper() for t in ERROR_TOKENS]:
            return 0
        return v
    return v  # 已經是數值/其他型別就保留

def clean_dataframe(df: pd.DataFrame):
    """
    對整張表逐格清洗：
    - 空白 / Error / #DIV/0! → 0
    - 其他值維持
    回傳 df_clean
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(normalize_cell_value)
    return df_clean

def can_be_all_numeric_after_clean(series: pd.Series):
    """
    檢查某欄是否「在補0處理之後，整欄都能轉成數字」。
    回傳:
      (is_all_numeric, numeric_series_float)
    """
    def normalize_numeric_like(x):
        if isinstance(x, str):
            return x.replace(",", "").strip()
        return x
    normalized = series.apply(normalize_numeric_like)
    numeric_series = pd.to_numeric(normalized, errors="coerce")
    all_numeric = not numeric_series.isna().any()
    return all_numeric, numeric_series.astype(float)

def get_pure_numeric_cols(df_clean: pd.DataFrame):
    """
    回傳所有「可完全當成數字」的欄位名稱清單 (target用)
    以及對應的 float series 給後續用不到，但如果要更高效可以保留 map。
    """
    numeric_cols = []
    for col in df_clean.columns:
        all_num, _ = can_be_all_numeric_after_clean(df_clean[col])
        if all_num:
            numeric_cols.append(col)
    return numeric_cols

def summarize_by_dimension(df_clean: pd.DataFrame,
                           target_col: str,
                           dim_col: str,
                           mode: str,
                           top_n: int = 5):
    """
    mode:
      - "contribution": 最高貢獻 (誰佔最多)
      - "loss": 最大虧損 (誰最賠錢)

    做法：
    1. 針對 target_col（必須是數值指標）進行 groupby(dim_col).sum()
    2. contribution:
       - 由大到小排序
       - 算佔比
    3. loss:
       - 只保留負值群組
       - 依虧損絕對值排序 (最負的排最前)
    """
    total_value = df_clean[target_col].astype(float).sum()

    group_sum = (
        df_clean
        .groupby(dim_col)[target_col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    )

    if mode == "contribution":
        ordered = group_sum.sort_values(ascending=False).head(top_n)
        out = ordered.reset_index()
        if total_value != 0:
            out["pct_of_total_%"] = (out[target_col] / total_value * 100).round(2)
        else:
            out["pct_of_total_%"] = 0.0
        return out, total_value

    elif mode == "loss":
        loss_only = group_sum[group_sum < 0]
        if loss_only.empty:
            return pd.DataFrame(columns=[dim_col, target_col, "loss_abs"]), total_value
        loss_sorted = (
            loss_only
            .to_frame(name=target_col)
            .assign(loss_abs=lambda x: x[target_col].abs())
            .sort_values("loss_abs", ascending=False)
            .head(top_n)
            .reset_index()
        )
        return loss_sorted, total_value

    else:
        raise ValueError("mode must be 'contribution' or 'loss'")

def filter_df_by_dim_value(df_clean: pd.DataFrame,
                           dim_col: str,
                           dim_value,
                           cols_show=None):
    """
    回傳該群組的原始列明細。維持原本的值（包括中文、代碼等等）。
    """
    sub = df_clean[df_clean[dim_col] == dim_value].copy()
    if cols_show is not None:
        sub = sub[cols_show]
    return sub

########################################################
# 1. Streamlit 介面
########################################################

st.title("🔍 Root Cause Mini Tool")

st.markdown("""
流程：
1. 上傳一張只有單一工作表的 Excel  
2. 系統會把空白 / Error / #DIV/0! 這些異常值補成 0  
3. Step 1 只能選「純數字欄位」當分析指標 (例如：損益、金額、數量...)  
4. Step 3 的分群維度可以是任何欄位，包含中文的客戶名稱、產品別、業務員等等  
5. 你可以看「最高貢獻」或「最大虧損」，並往下展開看到明細
""")

uploaded_file = st.file_uploader("上傳 Excel (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.stop()

# 嘗試讀取單一工作表
try:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    if len(sheet_names) != 1:
        st.error(f"偵測到 {len(sheet_names)} 個工作表，請只保留單一工作表再上傳。")
        st.stop()
    sheet_name = sheet_names[0]
    raw_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
except Exception as e:
    st.error(f"讀檔失敗：{e}")
    st.stop()

# 基本表頭檢查
if raw_df.columns.duplicated().any():
    st.error("表頭有重複欄位名稱，請修正後再上傳。")
    st.stop()

if any(col is None or str(col).strip() == "" for col in raw_df.columns):
    st.error("有欄位名稱是空白，請補齊後再上傳。")
    st.stop()

st.success(f"✅ 成功讀取工作表：{sheet_name}，共 {raw_df.shape[0]} 列 x {raw_df.shape[1]} 欄")

########################################################
# 清洗資料 (補0處理)
########################################################

df_clean = clean_dataframe(raw_df)
st.info("已自動處理空白 / Error / #DIV/0!：這些值都已補成 0。")

########################################################
# 取得可用的「純數字欄位」-> 只能用在 Step 1
########################################################

numeric_cols = get_pure_numeric_cols(df_clean)

if len(numeric_cols) == 0:
    st.error("目前偵測不到任何『全部可轉數字』的欄位，無法進行數值分析。請確認資料中是否有金額/數量欄。")
    st.stop()

########################################################
# Step 1. 選分析指標 (只能數字欄)
########################################################

st.subheader("Step 1. 請選要分析的指標欄位（只能選純數字欄）")
target_col = st.selectbox(
    "分析目標欄位 (例如：損益、退貨金額、報廢數量...)",
    numeric_cols
)
st.caption("這個欄位會被拿來計算：誰貢獻最多？誰賠最多？")

########################################################
# Step 2. 選分析模式
########################################################

st.subheader("Step 2. 請選分析模式")
mode_label = st.radio(
    "你想要看哪一類主因？",
    ["最高貢獻 (誰佔最多)", "最大虧損 (誰最賠錢)"],
    index=0,
    horizontal=True
)
internal_mode = "contribution" if "貢獻" in mode_label else "loss"

st.caption("""
- 最高貢獻：把這個指標加總後，哪個群組佔的金額/數量最大  
- 最大虧損：只看負值（虧損），誰最嚴重
""")

########################################################
# Step 3. 選分群維度 (可以是任何欄位，包含中文/英文/名稱)
########################################################

st.subheader("Step 3. 請選要分群檢視的欄位 (客戶 / 產品別 / 地區 / 業務員 / 工站 / 機台...)")

# 候選維度 = 全部欄位
all_possible_dims = list(df_clean.columns)

# 避免維度跟指標是同一欄（通常沒意義：按X分組X本身）
if target_col in all_possible_dims:
    all_possible_dims.remove(target_col)

if len(all_possible_dims) == 0:
    st.error("沒有可用的分群欄位可以切分（除了你選的指標欄位本身之外）。")
    st.stop()

dim_col = st.selectbox(
    "選一個欄位來分組，看誰是主因",
    all_possible_dims
)

########################################################
# Step 4. 產生主因表、選嫌疑群組、展開明細
########################################################

summary_df, total_value = summarize_by_dimension(
    df_clean=df_clean,
    target_col=target_col,
    dim_col=dim_col,
    mode=internal_mode,
    top_n=5
)

if internal_mode == "contribution":
    st.markdown(f"**總 `{target_col}` = {total_value:.2f}**")
    if summary_df.empty:
        st.warning("沒有資料可用來計算。")
    else:
        st.write(f"依「{dim_col}」分組後，`{target_col}` 最大的 Top 5：")
        st.dataframe(summary_df)

else:  # 最大虧損
    st.markdown(f"**整體 `{target_col}` 加總 = {total_value:.2f}**")
    if summary_df.empty:
        st.info(f"沒有任何群組是負值（沒有虧損）。")
    else:
        st.write(f"依「{dim_col}」分組後，最賠錢的 Top 5 (依虧損絕對值排序)：")
        st.dataframe(summary_df)

if summary_df.empty:
    st.stop()

st.subheader("往下展開 (明細檢查)")

# 讓使用者選出其中一個群組的值進一步看原始列
candidate_values = summary_df[dim_col].astype(str).tolist()

selected_val = st.selectbox(
    f"請選一個『{dim_col}』的值，往下展開看明細：",
    candidate_values
)

detail_df = filter_df_by_dim_value(df_clean, dim_col, selected_val, cols_show=None)

st.markdown("### 明細摘要")
group_sum = pd.to_numeric(detail_df[target_col], errors="coerce").fillna(0).sum()
pct = (group_sum / total_value * 100) if total_value != 0 else 0

if internal_mode == "contribution":
    st.markdown(f"- 你目前檢視的是 **{dim_col} = {selected_val}**")
    st.markdown(f"- 這個群組的 `{target_col}` 合計：**{group_sum:.2f}**  (占比 {pct:.2f}%)")
else:
    st.markdown(f"- 你目前檢視的是 **{dim_col} = {selected_val}** (虧損檢視)")
    st.markdown(f"- 這個群組的 `{target_col}` 合計：**{group_sum:.2f}**")
    st.markdown(f"- 虧損絕對值：**{abs(group_sum):.2f}**")

st.markdown("### 原始列資料")
st.dataframe(detail_df)
