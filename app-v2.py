import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Excel表格分析 / 根因分析", layout="wide")

########################################################
# 0. 資料清洗 / 數字欄判斷 / 群組計算
########################################################

ERROR_TOKENS = [
    "", " ", "  ", "\t",
    "#DIV/0!", "#DIV/0", "#VALUE!", "#REF!", "#NAME?", "#N/A",
    "#NUM!", "#NULL!", "ERROR", "Error", "error"
]

def normalize_cell_value(v):
    """空白 / Error / #DIV/0! 等 → 0，其它保留原值"""
    if pd.isna(v):
        return 0
    if isinstance(v, str):
        vs = v.strip()
        for bad in ERROR_TOKENS:
            if vs == bad or vs.upper() == bad.upper():
                return 0
        return v
    return v

def clean_dataframe(df: pd.DataFrame):
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(normalize_cell_value)
    return df_clean

def can_be_all_numeric_after_clean(series: pd.Series):
    """判斷整欄(補0後)是否都能轉數字"""
    def normalize_numeric_like(x):
        if isinstance(x, str):
            return x.replace(",", "").strip()
        return x
    normalized = series.apply(normalize_numeric_like)
    numeric_series = pd.to_numeric(normalized, errors="coerce")
    all_numeric = not numeric_series.isna().any()
    return all_numeric, numeric_series.astype(float)

def get_pure_numeric_cols(df_clean: pd.DataFrame):
    """找可完全轉成數字的欄位"""
    numeric_cols = []
    for col in df_clean.columns:
        all_num, _ = can_be_all_numeric_after_clean(df_clean[col])
        if all_num:
            numeric_cols.append(col)
    return numeric_cols

def summarize_by_dimension(df_clean, target_col, dim_col, mode, top_n=5):
    """
    用在第一頁(表格分析)
    mode:
      - contribution: 最高貢獻
      - loss: 最大虧損(負值)
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
    回傳該群組的原始列明細，字串比對避免型別不一。
    """
    mask = df_clean[dim_col].astype(str) == str(dim_value)
    sub = df_clean[mask].copy()
    if cols_show is not None:
        sub = sub[cols_show]
    return sub


########################################################
# 1. 根因分析樹 計算邏輯
########################################################

def apply_filters_to_df(df, filters: dict):
    """
    filters = {"地區": "北區", "產品": "A系列", ...}
    回傳符合所有條件的subset
    """
    sub = df.copy()
    for col_name, col_val in filters.items():
        if col_val == "__ALL__":
            continue
        sub = sub[sub[col_name].astype(str) == str(col_val)]
    return sub

def compute_top_groups_with_other(df_sub, target_col, split_dim, top_n=5):
    """
    在 df_sub 中使用 split_dim 分群，計算 target_col 合計
    1. 依絕對值(|kpi_sum|)排序
    2. Top N
    3. 其餘合併成 "其他"
    回傳 list[ dict ]:
      {
        "label": 群組名稱,
        "kpi_sum": float,
        "pct": group對parent的占比(%),
        "filters": { split_dim: value or "(其他)" },
        "is_other": bool
      }
    """
    total_here = pd.to_numeric(df_sub[target_col], errors="coerce").fillna(0).sum()

    grp = (
        df_sub.groupby(split_dim)[target_col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    )

    if grp.empty:
        return []

    grp_sorted = grp.reindex(grp.abs().sort_values(ascending=False).index)

    top_grp = grp_sorted.head(top_n)
    other_grp = grp_sorted.iloc[top_n:]

    rows = []
    for gval, ksum in top_grp.items():
        pct = 0.0 if total_here == 0 else (ksum / total_here * 100.0)
        rows.append({
            "label": str(gval),
            "kpi_sum": float(ksum),
            "pct": float(pct),
            "filters": {split_dim: str(gval)},
            "is_other": False
        })

    if len(other_grp) > 0:
        other_sum = other_grp.sum()
        other_pct = 0.0 if total_here == 0 else (other_sum / total_here * 100.0)
        rows.append({
            "label": "其他",
            "kpi_sum": float(other_sum),
            "pct": float(other_pct),
            "filters": {split_dim: "(其他)"},
            "is_other": True
        })

    return rows

def compute_dimension_spread(df_sub, target_col, dim_col):
    """
    該欄位能解釋KPI的程度(差異度): max(group_sum)-min(group_sum)
    """
    grp_series = (
        df_sub.groupby(dim_col)[target_col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    )
    if grp_series.empty:
        return 0.0
    return float(abs(grp_series.max() - grp_series.min()))

def rank_candidate_dims(df_sub, target_col, all_dims, used_dims):
    """
    針對目前的 parent subset，哪些欄位最值得拆？
    回傳 spread 大到小的欄位清單
    """
    scored = []
    for dim in all_dims:
        if dim == target_col:
            continue
        if dim in used_dims:
            continue
        spread = compute_dimension_spread(df_sub, target_col, dim)
        scored.append((dim, spread))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]


########################################################
# 2. 樹的 state (rca_layers)，以及樹視覺化
########################################################

def init_rca_layers():
    """
    rca_layers: list of layer dicts
    layer dict:
    {
      "parent_filters": {...},        # 例如 {}, 或 {"廠處":"H6"}
      "split_dim": "廠處",            # 這層用哪個欄位拆
      "groups": [
         {
           "label":"H6",
           "kpi_sum":-42728.81,
           "pct":37.43,
           "filters":{"廠處":"H6"},
           "is_other":False
         },
         ...
      ]
    }
    """
    if "rca_layers" not in st.session_state:
        st.session_state["rca_layers"] = []

def list_all_existing_parent_filters():
    """
    取得所有目前可當成「父節點」繼續往下拆的條件組合。
    包括：
    - 根 ({} = 全體)
    - rca_layers 裡每個 group node 的條件
    """
    results = []
    results.append(("整體 (無條件)", {}))

    for layer in st.session_state["rca_layers"]:
        base_filters = layer["parent_filters"]
        split_dim = layer["split_dim"]
        for g in layer["groups"]:
            fdict = {}
            fdict.update(base_filters)
            for k,v in g["filters"].items():
                if k == "__other__":
                    continue
                if k == split_dim:
                    fdict[k] = v
            if len(fdict)==0:
                disp_label = f"{split_dim}={g['label']} (全域?)"
            else:
                pairs_txt = ", ".join([f"{kk}={vv}" for kk,vv in fdict.items()])
                disp_label = pairs_txt
            if disp_label.strip()=="":
                disp_label = "(整體)"
            results.append((disp_label, fdict))

    unique = {}
    for (lab, fdict) in results:
        key = tuple(sorted(fdict.items()))
        unique[key] = (lab, fdict)

    final_list = list(unique.values())
    return final_list

def add_new_layer(df_clean, target_col, parent_filters, split_dim, top_n=5):
    """
    對 parent_filters 對應的subset，用 split_dim 分解 target_col → Top5+其他
    產生/更新一個 layer
    """
    df_sub = apply_filters_to_df(df_clean, parent_filters)
    groups = compute_top_groups_with_other(df_sub, target_col, split_dim, top_n=top_n)
    if len(groups) == 0:
        return

    replaced = False
    for lyr in st.session_state["rca_layers"]:
        if lyr["parent_filters"] == parent_filters and lyr["split_dim"] == split_dim:
            lyr["groups"] = groups
            replaced = True
            break

    if not replaced:
        st.session_state["rca_layers"].append({
            "parent_filters": parent_filters.copy(),
            "split_dim": split_dim,
            "groups": groups,
        })

def remove_layer(parent_filters, split_dim):
    """
    刪除一層展開
    """
    new_layers = []
    for lyr in st.session_state["rca_layers"]:
        if not (lyr["parent_filters"] == parent_filters and lyr["split_dim"] == split_dim):
            new_layers.append(lyr)
    st.session_state["rca_layers"] = new_layers


########################################################
# 2A. 把 rca_layers 轉成一條階層路徑 (levels)
########################################################

def value_color_class(val):
    """
    給數值一個 CSS class 方便上色
    """
    try:
        v = float(val)
    except:
        return "neutral"
    if v < 0:
        return "bad"
    elif v > 0:
        return "good"
    else:
        return "neutral"

def build_tree_path(df_clean, target_col):
    """
    依照使用者展開順序 (rca_layers append 順序)，
    產出層級描述，給前端 SmartArt 風格視覺化。
    levels = [
      {
        "title": "ROOT KPI",
        "subtitle": "全體資料",
        "nodes": [
            {"name":"淨利","value":..., "pct":"100.00%","color_class":"bad/good/neutral"}
        ]
      },
      {
        "title": "拆分欄位：廠處",
        "subtitle": "父條件: 全體",
        "nodes": [
            {"name":"H6","value":...,"pct":"37.43%","color_class":"bad"},
            ...
        ]
      },
      ...
    ]
    """
    levels = []

    # Root level
    full_total = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0).sum()
    root_node = {
        "name": target_col,
        "value": full_total,
        "pct": "100.00%",
        "color_class": value_color_class(full_total)
    }
    levels.append({
        "title": "ROOT KPI",
        "subtitle": "全體資料",
        "nodes": [root_node]
    })

    # 每個使用者展開過的 layer
    for layer in st.session_state["rca_layers"]:
        p_filters = layer["parent_filters"]
        split_dim = layer["split_dim"]
        groups   = layer["groups"]

        df_parent = apply_filters_to_df(df_clean, p_filters)
        parent_total = pd.to_numeric(df_parent[target_col], errors="coerce").fillna(0).sum()

        if len(p_filters)==0:
            pf_str = "全體"
        else:
            pf_str = ", ".join([f"{k}={v}" for k,v in p_filters.items()])

        node_list = []
        for g in groups:
            pct_parent = 0.0
            if parent_total != 0:
                pct_parent = g["kpi_sum"]/parent_total*100.0
            node_list.append({
                "name": g["label"],
                "value": g["kpi_sum"],
                "pct": f"{pct_parent:.2f}%",
                "color_class": value_color_class(g["kpi_sum"])
            })

        levels.append({
            "title": f"拆分欄位：{split_dim}",
            "subtitle": f"父條件: {pf_str}",
            "nodes": node_list
        })

    return levels


########################################################
# 2B. 把 levels 畫成 SmartArt 風格的 HTML/CSS
########################################################

def render_tree_html(levels):
    """
    將 build_tree_path() 的結果畫成樹：
    - 第一層顯示 KPI 總體
    - 後續每層是「拆分欄位：xxx」的一排 cards
    - flex 橫向排列孩子，並用細線連到上一層
    """
    css = """
    <style>
    .rca-wrapper{
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
        font-size:13px;
        color:#222;
    }
    .rca-level{
        display:flex;
        flex-direction:column;
        align-items:center;
        margin-bottom:32px;
        position:relative;
    }
    .rca-titlebox{
        text-align:center;
        margin-bottom:12px;
    }
    .rca-title{
        font-size:0.8rem;
        font-weight:600;
        color:#111;
        line-height:1.2;
    }
    .rca-sub{
        font-size:0.7rem;
        color:#666;
        line-height:1.2;
    }
    .rca-branch-row{
        display:flex;
        flex-wrap:wrap;
        justify-content:center;
        gap:16px;
        position:relative;
        padding:8px 0;
        min-height:80px;
    }
    .rca-branch-row-line{
        position:absolute;
        top:-16px;
        left:0;
        right:0;
        height:0;
        border-top:1px solid #999;
        z-index:-1;
    }
    .rca-connector-vertical{
        position:absolute;
        top:-16px;
        left:50%;
        width:0;
        height:16px;
        border-left:1px solid #999;
    }
    .rca-branch{
        position:relative;
        display:flex;
        flex-direction:column;
        align-items:center;
    }
    .rca-branch-connector{
        position:absolute;
        top:-16px;
        left:50%;
        width:1px;
        height:16px;
        border-left:1px solid #999;
    }
    .rca-node-card{
        min-width:120px;
        max-width:160px;
        background:#fff;
        border-radius:8px;
        box-shadow:0 2px 6px rgba(0,0,0,0.08);
        border:1px solid #ddd;
        padding:8px 10px;
        text-align:center;
        line-height:1.3;
        font-size:0.7rem;
    }
    .rca-node-name{
        font-weight:600;
        font-size:0.75rem;
        color:#222;
        margin-bottom:4px;
        word-wrap:break-word;
        word-break:break-all;
    }
    .rca-node-value{
        font-size:0.75rem;
        margin-bottom:2px;
        font-family:Consolas,Menlo,monospace;
        font-weight:600;
        color:#444;
    }
    .rca-node-value.bad{ color:#c62828; }
    .rca-node-value.good{ color:#2e7d32; }
    .rca-node-value.neutral{ color:#444; }
    .rca-node-pct{
        font-size:0.65rem;
        color:#666;
    }
    </style>
    """

    body = ['<div class="rca-wrapper">']

    for idx, level in enumerate(levels):
        title = level["title"]
        subtitle = level["subtitle"]
        nodes = level["nodes"]

        body.append('<div class="rca-level">')

        # Title box
        body.append('<div class="rca-titlebox">')
        body.append(f'<div class="rca-title">{title}</div>')
        body.append(f'<div class="rca-sub">{subtitle}</div>')
        body.append('</div>')

        # Row of children cards
        body.append('<div class="rca-branch-row">')
        # 畫連線(除了第0層以外)
        if idx > 0:
            body.append('<div class="rca-branch-row-line"></div>')
            body.append('<div class="rca-connector-vertical"></div>')

        for n in nodes:
            body.append('<div class="rca-branch">')
            if idx > 0:
                body.append('<div class="rca-branch-connector"></div>')
            body.append('<div class="rca-node-card">')
            body.append(f'<div class="rca-node-name">{n["name"]}</div>')
            body.append(
                f'<div class="rca-node-value {n["color_class"]}">{n["value"]:,.2f}</div>'
            )
            body.append(f'<div class="rca-node-pct">{n["pct"]}</div>')
            body.append('</div>')  # card
            body.append('</div>')  # branch

        body.append('</div>')  # branch-row
        body.append('</div>')  # level

    body.append('</div>')  # wrapper

    return css + "\n".join(body)


########################################################
# 3. 共用小UI元件
########################################################

def step_header_small(text):
    st.markdown(
        f"<div style='color:#666;font-size:0.9rem;font-weight:500;margin-top:1rem;'>{text}</div>",
        unsafe_allow_html=True
    )

def big_bold_label(text):
    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;'>{text}</div>",
        unsafe_allow_html=True
    )

def pick_one_from_grid_scrollable(
    options,
    current_value,
    key_prefix,
    columns_per_row=4,
    box_height_px=300
):
    chosen = current_value

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

    rows = [options[i:i+columns_per_row] for i in range(0, len(options), columns_per_row)]

    global_idx = 0
    for r_i, row_options in enumerate(rows):
        cols = st.columns(len(row_options))
        for c_i, opt in enumerate(row_options):
            btn_key = f"{key_prefix}_{global_idx}"
            if cols[c_i].button(
                opt,
                key=btn_key,
                use_container_width=True
            ):
                chosen = opt
            global_idx += 1

    st.markdown("</div>", unsafe_allow_html=True)

    return chosen


########################################################
# 4. 第一頁：Excel表格分析 (保持原本功能)
########################################################

def page_table_analysis():
    st.title("📊 Excel表格分析小工具")

    st.markdown("""
    ### 📘 使用說明
    流程：  
    Step 1. 上傳 Excel、選擇欲分析的工作表  
    ⚠ 注意：工作表第一列需為欄位名稱，且表格外請勿留雜資料，以免分析錯誤  
    系統會自動將空白 / Error / #DIV/0! 等異常值補成 0。  
    Step 2. 選分析指標 (只能數字欄)  
    Step 3. 選模式（最高貢獻 / 最大虧損）  
    Step 4. 選分群欄位  
    Step 5. 展開明細  
    Step 6. 下載明細  
    <br>
    <span style='color:red; font-weight:bold;'>
    資安聲明：此網站不會記錄使用者上傳的檔案，也不會記憶任何資訊
    </span>
    """, unsafe_allow_html=True)

    step_header_small("Step 1. 上傳 Excel、選擇欲分析的工作表")
    big_bold_label("上傳 Excel (.xls / .xlsx / .xlsm)")

    uploaded_file = st.file_uploader("", type=["xls","xlsx","xlsm"], key="page1_uploader")
    if uploaded_file is None:
        st.stop()

    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
    except Exception as e:
        st.error(f"讀檔失敗：{e}")
        st.stop()

    big_bold_label("請選擇要分析的工作表")
    chosen_sheet = st.selectbox(
        "選擇工作表(下拉式選單)",
        sheet_names,
        index=0,
        key="page1_sheet"
    )

    if "page1_last_sheet" not in st.session_state:
        st.session_state["page1_last_sheet"] = chosen_sheet
    if chosen_sheet != st.session_state["page1_last_sheet"]:
        for k in ["target_col", "dim_col", "selected_val"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["page1_last_sheet"] = chosen_sheet

    try:
        raw_df = pd.read_excel(uploaded_file, sheet_name=chosen_sheet, header=0)
    except Exception as e:
        st.error(f"讀取工作表失敗：{e}")
        st.stop()

    if raw_df.columns.duplicated().any():
        st.error("表頭有重複欄位名稱，請修正後再上傳。")
        st.stop()
    if any(col is None or str(col).strip()=="" for col in raw_df.columns):
        st.error("有欄位名稱是空白，請補齊後再上傳。")
        st.stop()

    st.success(f"✅ 成功讀取工作表：{chosen_sheet}，共 {raw_df.shape[0]} 列 × {raw_df.shape[1]} 欄")

    df_clean = clean_dataframe(raw_df)

    numeric_cols = get_pure_numeric_cols(df_clean)
    if not numeric_cols:
        st.error("未偵測到可完全轉成數字的欄位，請確認表格中有金額/數量欄位。")
        st.stop()

    # Step2 目標欄位
    step_header_small("Step 2. 選擇分析目標欄位 (純數字欄)")
    big_bold_label("請選擇欲分析之目標欄位（例如：損益、金額、報廢數量...)")

    st.session_state.setdefault("target_col", None)
    st.session_state["target_col"] = pick_one_from_grid_scrollable(
        options=numeric_cols,
        current_value=st.session_state["target_col"],
        key_prefix="targetcol_page1"
    )
    target_col = st.session_state["target_col"]

    if target_col is None:
        st.warning("請選一個分析指標欄位。")
        st.stop()

    col_sum = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0).sum()
    st.info(f"📊 `{target_col}` 欄位總合：{col_sum:,.2f}")

    # Step3 模式
    step_header_small("Step 3. 選分析模式")
    big_bold_label("你想要看哪一類主因？")

    mode_label = st.radio(
        "",
        ["最高貢獻 (誰佔最多-contribution)", "最大虧損 (誰最賠錢-loss)"],
        horizontal=True,
        key="page1_mode"
    )
    internal_mode = "contribution" if "貢獻" in mode_label else "loss"

    # Step4 分群欄位
    step_header_small("Step 4. 選分群欄位 (如：客戶、地區、業務員...)")
    big_bold_label("請選擇要分群的欄位")

    all_possible_dims = [c for c in df_clean.columns if c != target_col]
    if not all_possible_dims:
        st.error("無可用分群欄位。")
        st.stop()

    st.session_state.setdefault("dim_col", None)
    st.session_state["dim_col"] = pick_one_from_grid_scrollable(
        options=all_possible_dims,
        current_value=st.session_state["dim_col"],
        key_prefix="dimcol_page1"
    )
    dim_col = st.session_state["dim_col"]

    if dim_col is None:
        st.warning("請先選擇分群欄位。")
        st.stop()

    # Step5 展開明細
    summary_df, total_value = summarize_by_dimension(
        df_clean,
        target_col,
        dim_col,
        internal_mode
    )
    if summary_df.empty:
        st.warning("沒有資料可用來計算(例如沒有負數)。")
        st.stop()

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
        key_prefix="valsel_page1",
        box_height_px=200
    )
    selected_val = st.session_state["selected_val"]

    if selected_val is None:
        st.warning("請先選一個群組值。")
        st.stop()

    detail_df = filter_df_by_dim_value(df_clean, dim_col, selected_val)

    group_sum = pd.to_numeric(detail_df[target_col], errors="coerce").fillna(0).sum()
    pct = (group_sum / total_value * 100) if total_value != 0 else 0

    st.markdown(f"**目前檢視：** {dim_col} = {selected_val}")
    st.markdown(f"**{target_col} 合計：** {group_sum:.2f}（占比 {pct:.2f}%）")

    st.dataframe(detail_df)

    # Step6 匯出
    step_header_small("Step 6. 下載明細")
    big_bold_label("匯出這次分析結果（Excel）")

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
        metadata.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=0,
            index=False
        )

        start_row = len(metadata) + 2
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


########################################################
# 5. 第二頁：SmartArt風格 根因分析樹 (HTML/CSS)
########################################################

def page_root_cause_tree():
    st.markdown("#### 🌳 根因分析樹 (SmartArt階層風格)")

    st.markdown(
        "<div style='font-size:0.8rem;color:#444;margin-bottom:0.5rem;'>"
        "上傳檔案 → 選 KPI → 選要在哪個節點往下拆 → 選欄位，"
        "系統會在下方畫出一層一層往下長的階層圖（類似 PowerPoint SmartArt 組織圖）。<br>"
        "右邊可以新增下一層，或刪掉某一層，或全部清除。"
        "</div>",
        unsafe_allow_html=True
    )

    init_rca_layers()

    # 上傳檔案
    uploaded_file = st.file_uploader(
        "上傳 Excel (.xls / .xlsx / .xlsm) (此頁只讀第一個工作表)",
        type=["xls","xlsx","xlsm"],
        key="page2_uploader"
    )
    if uploaded_file is None:
        st.stop()

    try:
        xls = pd.ExcelFile(uploaded_file)
        first_sheet = xls.sheet_names[0]
        df_raw = pd.read_excel(uploaded_file, sheet_name=first_sheet, header=0)
    except Exception as e:
        st.error(f"讀檔失敗：{e}")
        st.stop()

    # 表頭檢查
    if df_raw.columns.duplicated().any():
        st.error("表頭有重複欄位名稱，請修正後再上傳。")
        st.stop()
    if any(col is None or str(col).strip()=="" for col in df_raw.columns):
        st.error("有欄位名稱是空白，請補齊後再上傳。")
        st.stop()

    df_clean = clean_dataframe(df_raw)

    # KPI 欄
    numeric_cols = get_pure_numeric_cols(df_clean)
    if not numeric_cols:
        st.error("未偵測到可完全轉成數字的欄位(例如金額/淨利/數量)。")
        st.stop()

    target_col = st.selectbox(
        "請選擇要分析的 KPI (數值欄位)",
        numeric_cols,
        key="page2_target_col"
    )

    # 可拆分的維度欄位
    all_dims = [c for c in df_clean.columns if c != target_col]

    # 排版：左=樹；右=控制面板
    main_col, side_col = st.columns([4, 1.6], gap="large")

    # 右側：互動控制
    with side_col:
        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#000;"
            "margin-bottom:0.5rem;border-bottom:1px solid #ccc;'>展開 / 維護樹</div>",
            unsafe_allow_html=True
        )

        # 可展開的 "父節點"
        parent_choices = list_all_existing_parent_filters()
        parent_labels = [lab for (lab, fdict) in parent_choices]
        sel_parent_label = st.selectbox(
            "我要在哪個節點下面新增下一層？",
            options=parent_labels,
            key="rca_parent_select"
        )

        parent_filters = {}
        for (lab, fdict) in parent_choices:
            if lab == sel_parent_label:
                parent_filters = fdict.copy()
                break

        # 在該父節點下，哪些欄位最值得拆？
        df_parent_sub = apply_filters_to_df(df_clean, parent_filters)

        used_dims_here = [
            lyr["split_dim"]
            for lyr in st.session_state["rca_layers"]
            if lyr["parent_filters"] == parent_filters
        ]

        ranked_dims = rank_candidate_dims(
            df_parent_sub, target_col, all_dims, used_dims_here
        )

        if len(ranked_dims)==0:
            st.info("在這個節點下，沒有更多可拆的欄位 (或都拆過了)")
        else:
            sel_split_dim = st.selectbox(
                "下一層要用哪個欄位分解？(依影響力排序)",
                options=ranked_dims,
                key="rca_split_dim_select"
            )

            if st.button("➕ 加到樹裡 (展開下一層)"):
                add_new_layer(
                    df_clean=df_clean,
                    target_col=target_col,
                    parent_filters=parent_filters,
                    split_dim=sel_split_dim,
                    top_n=5
                )
                st.rerun()

        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#000;"
            "margin:1rem 0 0.5rem;border-bottom:1px solid #ccc;'>刪除某層拆解</div>",
            unsafe_allow_html=True
        )

        if len(st.session_state["rca_layers"])==0:
            st.write("目前沒有任何展開的層")
        else:
            layer_labels = []
            for i,lyr in enumerate(st.session_state["rca_layers"]):
                pf = lyr["parent_filters"]
                pf_txt = "全體" if len(pf)==0 else ", ".join([f"{k}={v}" for k,v in pf.items()])
                layer_labels.append(f"{i}. 父條件[{pf_txt}] → 拆分欄位:{lyr['split_dim']}")

            sel_layer_to_remove = st.selectbox(
                "選擇要刪除的層：",
                options=layer_labels,
                key="rca_remove_layer_select"
            )

            if st.button("🗑 刪除此層 (不會刪掉其他層)"):
                idx_to_remove = layer_labels.index(sel_layer_to_remove)
                lyr = st.session_state["rca_layers"][idx_to_remove]
                remove_layer(
                    parent_filters=lyr["parent_filters"],
                    split_dim=lyr["split_dim"]
                )
                st.rerun()

        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#000;"
            "margin:1rem 0 0.5rem;border-bottom:1px solid #ccc;'>全部清除</div>",
            unsafe_allow_html=True
        )

        if st.button("💣 清空整棵樹 (回到只有KPI)"):
            st.session_state["rca_layers"] = []
            st.rerun()

    # 左側：樹形 SmartArt 風格視覺
    with main_col:
        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#000;"
            "margin-bottom:0.5rem;'>目前的根因樹 (SmartArt風格)</div>",
            unsafe_allow_html=True
        )

        levels = build_tree_path(df_clean, target_col)
        html_tree = render_tree_html(levels)

        st.markdown(html_tree, unsafe_allow_html=True)

        st.caption(
            "說明：每一層是你展開的下一層分解。卡片顯示群組名稱、該群組的 KPI 合計、"
            "以及該群組對上層(父節點)的占比。紅色=虧損最大，綠色=獲利。"
        )


########################################################
# 6. 頂部超扁導覽列
########################################################

def render_top_nav():
    st.markdown(
        """
        <style>
        div[data-baseweb="radio"] > div {
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    nav_col = st.container()
    with nav_col:
        c1, c2 = st.columns([0.12, 0.88])
        with c1:
            st.markdown(
                "<div style='font-size:0.7rem; color:#666; line-height:1; padding-top:2px;'>功能選單</div>",
                unsafe_allow_html=True
            )
        with c2:
            st.radio(
                label="功能選單",
                options=["📊 Excel表格分析", "🌳 根因分析樹"],
                horizontal=True,
                key="main_page_selector",
                label_visibility="collapsed",
            )
    st.markdown(
        """
        <div style="
            border-top:1px solid #CCC;
            margin-top:4px;
            margin-bottom:8px;
        "></div>
        """,
        unsafe_allow_html=True
    )

    return st.session_state["main_page_selector"]


########################################################
# 7. 主流程
########################################################

if "main_page_selector" not in st.session_state:
    st.session_state["main_page_selector"] = "📊 Excel表格分析"

current_page = render_top_nav()

if current_page == "📊 Excel表格分析":
    page_table_analysis()
else:
    page_root_cause_tree()
