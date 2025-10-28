import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Excel表格分析小工具", layout="wide")

########################################################
# 0. 公用邏輯
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
        for bad in ERROR_TOKENS:
            if vs == bad:
                return 0
        if vs.upper() in [t.upper() for t in ERROR_TOKENS]:
            return 0
        return v
    return v

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
    回傳所有「可完全當成數字」的欄位名稱清單 (target 用)
    """
    numeric_cols = []
    for col in df_clean.columns:
        all_num, _ = can_be_all_numeric_after_clean(df_clean[col])
        if all_num:
            numeric_cols.append(col)
    return numeric_cols

def summarize_by_dimension(df_clean, target_col, dim_col, mode, top_n=5):
    """
    根據指定的 target_col (數字指標) 與分群欄位 dim_col
    回傳 Top5 群組彙總表 & 整體總和
    mode:
      - "contribution": 最高貢獻 (誰佔最多)
      - "loss": 最大虧損 (誰最賠錢，只看負值)
    """
    total_value = df_clean[target_col].astype(float).sum()
    group_sum = (
        df_clean.groupby(dim_col)[target_col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    )

    if mode == "contribution":
        ordered = group_sum.sort_values(ascending=False).head(top_n)
        out = ordered.reset_index()
        out["pct_of_total_%"] = (
            out[target_col] / total_value * 100
        ).round(2) if total_value != 0 else 0.0
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

    重要修正：
    為避免型別不一致（例如 df_clean[dim_col] 是數字 10112，但使用者在 UI 選到的是字串 "10112"），
    我們在比對時把雙方都轉成字串再比。
    """
    mask = df_clean[dim_col].astype(str) == str(dim_value)
    sub = df_clean[mask].copy()

    if cols_show is not None:
        sub = sub[cols_show]
    return sub


########################################################
# UI helpers
########################################################
def step_header_small(step_text):
    """
    小字灰色 Step 說明
    """
    st.markdown(
        f"<div style='color:#666;font-size:0.9rem;font-weight:500;margin-top:1rem;'>{step_text}</div>",
        unsafe_allow_html=True
    )

def big_bold_label(label_text):
    """
    粗體較大的操作說明
    """
    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;'>{label_text}</div>",
        unsafe_allow_html=True
    )

def pick_one_from_grid_scrollable(
    options,
    current_value,
    key_prefix,
    columns_per_row=4,
    box_height_px=300
):
    """
    用九宮格 / 卡片式按鈕 + 可捲動區塊 來選一個值
    options: list[str] 選項
    current_value: 目前選到的值 (或 None)
    key_prefix: 用來組 button 的 key
    columns_per_row: 一列擺幾個按鈕
    box_height_px: 捲軸容器高度(px)
    回傳: 最新被選到的值
    """
    chosen = current_value

    # 捲軸外框
    st.markdown(
        f"""
        <div style="
            max-height:{box_height_px}px;
            overflow-y:auto;
            border:1px solid #CCC;
            border-radius:8px;
            padding:8px;
            background-color:#fafafa;
        ">
        """,
        unsafe_allow_html=True
    )

    # 切成多列
    rows = [options[i:i+columns_per_row] for i in range(0, len(options), columns_per_row)]

    for r_i, row_options in enumerate(rows):
        cols = st.columns(len(row_options))
        for c_i, opt in enumerate(row_options):
            is_selected = (opt == chosen)
            label_text = f"✅ {opt}" if is_selected else opt
            if cols[c_i].button(
                label_text,
                key=f"{key_prefix}_{r_i}_{c_i}_{opt}",
                use_container_width=True
            ):
                chosen = opt

    st.markdown("</div>", unsafe_allow_html=True)
    return chosen


########################################################
# Step 1. 上傳 Excel 並選工作表
########################################################
st.title("📊 Excel表格分析小工具")

st.markdown("""
### 📘 使用說明
**流程：**  
**Step 1.** 上傳 Excel、選擇欲分析的工作表  
　⚠️ **注意：工作表第一列需為欄位名稱，且表格外請勿留雜資料，以免分析錯誤**  
　📌 系統會自動將空白 / Error / #DIV/0! 等異常值補成 0。  
**Step 2.** 選分析指標 (只能數字欄)  
**Step 3.** 選模式（最高貢獻 / 最大虧損）  
**Step 4.** 選分群欄位  
**Step 5.** 展開明細  
**Step 6.** 下載明細  

資安聲明:
**此網站不會記錄使用者上傳的檔案，也不會記憶任何資訊**

""")

uploaded_file = st.file_uploader(
    "上傳 Excel (.xls / .xlsx / .xlsm)",
    type=["xls", "xlsx", "xlsm"]
)
if uploaded_file is None:
    st.stop()

# 嘗試解析出所有工作表名稱
try:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
except Exception as e:
    st.error(f"讀檔失敗：{e}")
    st.stop()

step_header_small("Step 1. 上傳 Excel、選擇欲分析的工作表")
big_bold_label("請選擇要分析的工作表")

chosen_sheet = st.selectbox("選擇工作表", sheet_names, index=0)

# 切換工作表時，重置後續分析狀態，避免沿用舊欄位
if "last_sheet" not in st.session_state:
    st.session_state["last_sheet"] = chosen_sheet
if chosen_sheet != st.session_state["last_sheet"]:
    for k in ["target_col", "dim_col", "selected_val"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["last_sheet"] = chosen_sheet

# 讀取使用者選的 sheet
try:
    raw_df = pd.read_excel(uploaded_file, sheet_name=chosen_sheet, header=0)
except Exception as e:
    st.error(f"讀取工作表失敗：{e}")
    st.stop()

# 表頭基本檢查
if raw_df.columns.duplicated().any():
    st.error("表頭有重複欄位名稱，請修正後再上傳。")
    st.stop()

if any(col is None or str(col).strip() == "" for col in raw_df.columns):
    st.error("有欄位名稱是空白，請補齊後再上傳。")
    st.stop()

st.success(f"✅ 成功讀取工作表：{chosen_sheet}，共 {raw_df.shape[0]} 列 × {raw_df.shape[1]} 欄")

# 清洗資料（把空白/錯誤值補 0）
df_clean = clean_dataframe(raw_df)

# 找出可當作數字的欄位，這些欄位才能當分析指標
numeric_cols = get_pure_numeric_cols(df_clean)
if not numeric_cols:
    st.error("未偵測到可完全轉成數字的欄位，請確認表格中有金額/數量欄位。")
    st.stop()


########################################################
# Step 2. 選分析指標欄位 (純數字欄)
########################################################
step_header_small("Step 2. 選分析指標欄位 (純數字欄)")
big_bold_label("分析目標欄位（例如：損益、金額、報廢數量...)")

st.session_state.setdefault("target_col", None)
st.session_state["target_col"] = pick_one_from_grid_scrollable(
    options=numeric_cols,
    current_value=st.session_state["target_col"],
    key_prefix="targetcol"
)
target_col = st.session_state["target_col"]

if target_col is None:
    st.warning("請選一個分析指標欄位。")
    st.stop()


########################################################
# Step 3. 選分析模式
########################################################
step_header_small("Step 3. 選分析模式")
big_bold_label("你想要看哪一類主因？")

mode_label = st.radio(
    "",
    ["最高貢獻 (誰佔最多)", "最大虧損 (誰最賠錢)"],
    horizontal=True
)
internal_mode = "contribution" if "貢獻" in mode_label else "loss"


########################################################
# Step 4. 選分群欄位
########################################################
step_header_small("Step 4. 選分群欄位 (如：客戶、地區、業務員...)")
big_bold_label("請選擇要分群的欄位")

# 分群候選 = 全部欄位 - 目標指標欄位本身
all_possible_dims = [c for c in df_clean.columns if c != target_col]
if not all_possible_dims:
    st.error("無可用分群欄位。")
    st.stop()

st.session_state.setdefault("dim_col", None)
st.session_state["dim_col"] = pick_one_from_grid_scrollable(
    options=all_possible_dims,
    current_value=st.session_state["dim_col"],
    key_prefix="dimcol"
)
dim_col = st.session_state["dim_col"]

if dim_col is None:
    st.warning("請先選擇分群欄位。")
    st.stop()


########################################################
# Step 5. 展開明細
########################################################
summary_df, total_value = summarize_by_dimension(
    df_clean,
    target_col,
    dim_col,
    internal_mode
)

if summary_df.empty:
    st.warning("沒有資料可用來計算ex.資料沒有負數。")
    st.stop()

# 顯示 Top5
if internal_mode == "contribution":
    st.write(f"依「{dim_col}」分組後，`{target_col}` 最大的 Top 5：")
else:
    st.write(f"依「{dim_col}」分組後，最賠錢的 Top 5：")

st.dataframe(summary_df)

step_header_small("Step 5. 展開明細")
big_bold_label(f"請選一個『{dim_col}』的值檢視明細")

candidate_values = summary_df[dim_col].astype(str).tolist()

st.session_state.setdefault("selected_val", None)
st.session_state["selected_val"] = pick_one_from_grid_scrollable(
    options=candidate_values,
    current_value=st.session_state["selected_val"],
    key_prefix="valsel",
    box_height_px=200
)
selected_val = st.session_state["selected_val"]

if selected_val is None:
    st.warning("請先選一個群組值。")
    st.stop()

# 這裡用字串比對，避免 "10112"(str) vs 10112(int) 對不到
detail_df = filter_df_by_dim_value(df_clean, dim_col, selected_val)

group_sum = pd.to_numeric(detail_df[target_col], errors="coerce").fillna(0).sum()
pct = (group_sum / total_value * 100) if total_value != 0 else 0

st.markdown(f"**目前檢視：** {dim_col} = {selected_val}")
st.markdown(f"**{target_col} 合計：** {group_sum:.2f}（占比 {pct:.2f}%）")

st.dataframe(detail_df)


########################################################
# Step 6. 下載明細
########################################################
step_header_small("Step 6. 下載明細")
big_bold_label("匯出這次分析結果（Excel）")

# 將分析條件與彙總資訊整理成 DataFrame（第一列欄名、第二列值）
metadata = pd.DataFrame([{
    "timestamp_utc": datetime.utcnow().isoformat(),
    "source_file": uploaded_file.name,
    "sheet": chosen_sheet,
    "target_col": target_col,
    "mode": internal_mode,
    "dim_col": dim_col,
    "selected_val": selected_val,
    "group_sum": group_sum,
    "pct_of_total(%)": pct,
}])

output = io.BytesIO()

with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    sheet_name = "分析結果"

    # 先寫 metadata，第一列是欄名，第二列是值
    metadata.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=0,  # 第一列
        index=False
    )

    # 留一列空白（第3列）
    start_row = len(metadata) + 2  # 2 列 metadata + 1 列空白 = 第4列開始

    # 從第4列開始寫明細表格
    detail_df.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        index=False
    )

st.download_button(
    label="⬇ 下載 Excel 明細",
    data=output.getvalue(),
    file_name=f"分析結果_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
