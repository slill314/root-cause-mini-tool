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
        ordered = group_sum.sort_values(ascending=False)
    elif mode == "loss":
        ordered = group_sum.reindex(group_sum.sort_values(key=lambda x: x.abs(), ascending=False).index)
    else:
        ordered = group_sum

    top_only = ordered.head(top_n)
    other_only = ordered.iloc[top_n:]

    main_rows = top_only.reset_index()
    main_rows.columns = [dim_col, target_col]

    main_rows["pct_of_total_%"] = (
        main_rows[target_col] / total_value * 100
    ).round(2) if total_value != 0 else 0.0

    if len(other_only) > 0:
        other_sum = other_only.sum()
        other_pct = (other_sum / total_value * 100).round(2) if total_value != 0 else 0.0
        other_row = pd.DataFrame([{
            dim_col: "其他",
            target_col: other_sum,
            "pct_of_total_%": other_pct
        }])
        main_rows = pd.concat([main_rows, other_row], ignore_index=True)

    return main_rows, total_value


def filter_df_by_dim_value(df_clean: pd.DataFrame,
                           dim_col: str,
                           dim_value,
                           cols_show=None):
    mask = df_clean[dim_col].astype(str) == str(dim_value)
    sub = df_clean[mask].copy()
    if cols_show is not None:
        sub = sub[cols_show]
    return sub


########################################################
# 1. 根因分析樹 計算邏輯
########################################################

def apply_filters_to_df(df, filters: dict):
    sub = df.copy()
    for col_name, col_val in filters.items():
        if col_val == "__ALL__":
            continue
        sub = sub[sub[col_name].astype(str) == str(col_val)]
    return sub

def compute_top_groups_with_other(df_sub, target_col, split_dim, top_n=5):
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
        rows.append({ "label": str(gval), "kpi_sum": float(ksum), "pct": float(pct), "filters": {split_dim: str(gval)}, "is_other": False })

    if len(other_grp) > 0:
        other_sum = other_grp.sum()
        other_pct = 0.0 if total_here == 0 else (other_sum / total_here * 100.0)
        rows.append({ "label": "其他", "kpi_sum": float(other_sum), "pct": float(other_pct), "filters": {split_dim: "(其他)"}, "is_other": True })
    return rows

def compute_dimension_spread(df_sub, target_col, dim_col):
    grp_series = (
        df_sub.groupby(dim_col)[target_col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    )
    if grp_series.empty:
        return 0.0
    return float(abs(grp_series.max() - grp_series.min()))

def rank_candidate_dims(df_sub, target_col, all_dims, used_dims):
    scored = []
    for dim in all_dims:
        if dim == target_col or dim in used_dims:
            continue
        spread = compute_dimension_spread(df_sub, target_col, dim)
        scored.append((dim, spread))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

########################################################
# 2. 樹的 state (rca_layers) + 我們的新邏輯：單一路徑下鑽
########################################################

def init_rca_layers():
    if "rca_layers" not in st.session_state:
        st.session_state["rca_layers"] = []

def build_parent_candidates_from_tail():
    if len(st.session_state.get("rca_layers", [])) == 0:
        return [("全體 (無條件)", {})]
    
    tail = st.session_state["rca_layers"][-1]
    tail_parent_filters = tail["parent_filters"]
    split_dim = tail["split_dim"]

    out = []
    for g in tail["groups"]:
        fdict = tail_parent_filters.copy()
        fdict[split_dim] = g["label"]
        label_txt = ", ".join([f"{k}={v}" for k, v in fdict.items()])
        out.append((label_txt, fdict))

    unique_map = {tuple(sorted(fdict.items())): (lab, fdict) for lab, fdict in out}
    return list(unique_map.values())

def add_new_layer(df_clean, target_col, parent_filters, split_dim, top_n=5):
    valid_candidates = build_parent_candidates_from_tail()
    valid_filter_sets = [tuple(sorted(f.items())) for _, f in valid_candidates]
    if tuple(sorted(parent_filters.items())) not in valid_filter_sets:
        return

    df_sub = apply_filters_to_df(df_clean, parent_filters)
    groups = compute_top_groups_with_other(df_sub, target_col, split_dim, top_n=top_n)
    if not groups:
        return

    st.session_state["rca_layers"].append({
        "parent_filters": parent_filters.copy(),
        "split_dim": split_dim,
        "groups": groups,
    })

def remove_layer_by_index(idx_to_remove: int):
    if 0 <= idx_to_remove < len(st.session_state.get("rca_layers", [])):
        st.session_state["rca_layers"] = st.session_state["rca_layers"][:idx_to_remove]


########################################################
# 2B. 將樹渲染為垂直分解佈局的 HTML/CSS
########################################################

def value_color_class(val):
    try:
        v = float(val)
        if v < 0: return "bad"
        if v > 0: return "good"
        return "neutral"
    except (ValueError, TypeError):
        return "neutral"

def render_rca_tree_vertical(df_clean, target_col):
    # The wrapper padding for each node card. This value is used in CSS calc()
    node_padding_px = 10 
    
    css = f"""
    <style>
    .rca-vertical-container {{
        display: flex; flex-direction: column; align-items: center;
        padding: 20px 10px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    .rca-level-row {{
        display: flex; justify-content: center; align-items: flex-start;
        flex-wrap: nowrap; padding: 30px 0; position: relative;
    }}
    .rca-level-row:not(:last-child) {{ margin-bottom: 20px; }}
    .level-title-wrapper {{ text-align: center; }}
    .level-title {{ font-size: 0.8rem; font-weight: 600; color: #111; margin-bottom: 4px; }}
    .level-subtitle {{ font-size: 0.7rem; color: #666; font-weight: normal; margin-bottom: 15px;}}

    .rca-node-wrapper {{ 
        padding: 0 {node_padding_px}px; 
        position: relative; 
    }}
    .rca-node-card {{
        min-width: 140px; max-width: 180px;
        background: #fff; border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); border: 1px solid #ddd;
        padding: 8px 10px;
        line-height: 1.3; font-size: 0.7rem; text-align: center;
        transition: all 0.2s ease; cursor: default;
        position: relative; /* Needed for z-index to work */
        z-index: 10;
    }}
    .rca-node-name {{ font-weight: 600; font-size: 0.75rem; color: #222; margin-bottom: 4px; word-wrap: break-word; word-break: break-all; }}
    .rca-node-value {{ font-size: 0.75rem; margin-bottom: 2px; font-family: Consolas, Menlo, monospace; font-weight: 600; }}
    .rca-node-value.bad {{ color: #c62828; }}
    .rca-node-value.good {{ color: #2e7d32; }}
    .rca-node-value.neutral {{ color: #444; }}
    .rca-node-pct {{ font-size: 0.65rem; color: #666; }}

    /* --- 高亮與連接線 (更穩健的版本) --- */
    .rca-node-card.active-parent {{
        border-color: #007bff; border-width: 2px;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2);
    }}
    /* 從高亮父節點出發的向下的線 */
    .rca-node-card.active-parent::after {{
        content: ''; position: absolute;
        left: 50%; bottom: -32px;
        transform: translateX(-50%);
        width: 2px; height: 30px;
        background-color: #007bff;
    }}
    /* 每個子節點向上的線 */
    .rca-node-wrapper.is-child::before {{
        content: ''; position: absolute;
        left: 50%; top: -30px;
        transform: translateX(-50%);
        width: 2px; height: 30px;
        background-color: #007bff;
    }}
    /* 水平線段 (使用 calc 和 負 margin/position 來跨越 padding) */
    .rca-node-wrapper.is-child::after {{
        content: ''; position: absolute;
        top: -30px; height: 2px; background-color: #007bff;
        /* Default for middle nodes: stretch across the entire wrapper, including padding */
        left: -{node_padding_px}px;
        width: calc(100% + {2 * node_padding_px}px);
    }}
    /* 第一個子節點的水平線只向右 */
    .rca-node-wrapper.is-child:first-child::after {{
        left: 50%;
        width: calc(50% + {node_padding_px}px);
    }}
    /* 最後一個子節點的水平線只向左 */
    .rca-node-wrapper.is-child:last-child::after {{
        left: -{node_padding_px}px;
        width: calc(50% + {node_padding_px}px);
    }}
    /* 只有一個子節點時，不需要水平線 */
    .rca-node-wrapper.is-child:only-child::after {{
        display: none;
    }}
    </style>
    """
    
    html_rows = []
    
    active_path_names = [target_col]
    layers = st.session_state.get("rca_layers", [])
    for i, current_layer in enumerate(layers):
        if i + 1 < len(layers):
            next_layer = layers[i+1]
            choice_made = next_layer["parent_filters"].get(current_layer["split_dim"])
            if choice_made:
                active_path_names.append(choice_made)

    full_total = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0).sum()
    is_root_active = len(layers) > 0
    root_active_class = "active-parent" if is_root_active else ""
    
    root_html = '<div class="rca-level-row">'
    root_html += '<div class="level-title-wrapper">'
    root_html += '<div class="level-title">ROOT KPI</div><div class="level-subtitle">全體資料</div>'
    root_html += f'<div class="rca-node-wrapper"><div class="rca-node-card {root_active_class}">'
    root_html += f'<div class="rca-node-name">{target_col}</div>'
    root_html += f'<div class="rca-node-value {value_color_class(full_total)}">{full_total:,.2f}</div>'
    root_html += f'<div class="rca-node-pct">100.00%</div>'
    root_html += '</div></div></div></div>'
    html_rows.append(root_html)

    df_parent = df_clean
    for i, layer in enumerate(layers):
        parent_total = pd.to_numeric(df_parent[target_col], errors="coerce").fillna(0).sum()
        pf_str = ", ".join([f"{k}={v}" for k, v in layer["parent_filters"].items()])
        
        children_wrapper = '<div class="level-title-wrapper">'
        children_wrapper += f'<div class="level-title">拆分欄位：{layer["split_dim"]}</div>'
        children_wrapper += f'<div class="level-subtitle">父條件: {pf_str}</div>'
        
        is_last_layer = (i == len(layers) - 1)
        
        children_cards_html = '<div class="rca-level-row">'
        for g in layer["groups"]:
            pct_parent = 0.0 if parent_total == 0 else (g["kpi_sum"] / parent_total * 100.0)
            
            is_active_parent = not is_last_layer and g["label"] == active_path_names[i+1]
            active_class = "active-parent" if is_active_parent else ""

            children_cards_html += '<div class="rca-node-wrapper is-child">'
            children_cards_html += f'<div class="rca-node-card {active_class}">'
            children_cards_html += f'<div class="rca-node-name">{g["label"]}</div>'
            children_cards_html += f'<div class="rca-node-value {value_color_class(g["kpi_sum"])}">{g["kpi_sum"]:,.2f}</div>'
            children_cards_html += f'<div class="rca-node-pct">{pct_parent:.2f}%</div>'
            children_cards_html += '</div></div>'
        
        children_cards_html += '</div>'
        children_wrapper += children_cards_html + '</div>'
        html_rows.append(children_wrapper)
        
        df_parent = apply_filters_to_df(df_clean, layer["parent_filters"])

    return css + f'<div class="rca-vertical-container">{"".join(html_rows)}</div>'


########################################################
# 3. 共用小UI元件
########################################################
def step_header_small(text):
    st.markdown(f"<div style='color:#666;font-size:0.9rem;font-weight:500;margin-top:1rem;'>{text}</div>", unsafe_allow_html=True)

def big_bold_label(text):
    st.markdown(f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;'>{text}</div>", unsafe_allow_html=True)

def pick_one_from_grid_scrollable(options, current_value, key_prefix, columns_per_row=4, box_height_px=300):
    chosen = current_value
    st.markdown(f'<div style="max-height:{box_height_px}px; overflow-y:auto; border:1px solid #CCC; border-radius:8px; padding:8px; background-color:#fafafa;">', unsafe_allow_html=True)
    rows = [options[i:i+columns_per_row] for i in range(0, len(options), columns_per_row)]
    for r_i, row_options in enumerate(rows):
        cols = st.columns(len(row_options))
        for c_i, opt in enumerate(row_options):
            if cols[c_i].button(opt, key=f"{key_prefix}_{r_i}_{c_i}", use_container_width=True):
                chosen = opt
    st.markdown("</div>", unsafe_allow_html=True)
    return chosen

########################################################
# 4. 第一頁：Excel表格分析 (使用者提供版本)
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
# 5. 第二頁：SmartArt風格 根因分析樹
########################################################
def page_root_cause_tree():
    st.markdown("#### 🌳 根因分析樹 (垂直分解風格)")
    st.markdown("<div style='font-size:0.8rem;color:#444;margin-bottom:0.5rem;'>上傳檔案 → 選 KPI → 從『目前最後一層』指定某個節點往下拆 → 選欄位。<span style='color:red; font-weight:bold;'>
    資安聲明：此網站不會記錄使用者上傳的檔案，也不會記憶任何資訊
    </span></div>", unsafe_allow_html=True)

    init_rca_layers()

    uploaded_file = st.file_uploader("上傳 Excel (.xls / .xlsx / .xlsm) (此頁只讀第一個工作表)", type=["xls","xlsx","xlsm"], key="page2_uploader")
    if uploaded_file is None: st.stop()

    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=0)
    except Exception as e:
        st.error(f"讀檔失敗：{e}")
        st.stop()
    
    if df_raw.columns.duplicated().any() or any(col is None or str(col).strip()=="" for col in df_raw.columns):
        st.error("表頭有重複或空白的欄位名稱，請修正後再上傳。")
        st.stop()

    df_clean = clean_dataframe(df_raw)
    all_numeric_cols = get_pure_numeric_cols(df_clean)
    if not all_numeric_cols:
        st.error("未偵測到可完全轉成數字的欄位。")
        st.stop()

    used_dims_in_tree = {layer["split_dim"] for layer in st.session_state.get("rca_layers", [])}
    available_kpis = [col for col in all_numeric_cols if col not in used_dims_in_tree]
    if not available_kpis:
        st.error("所有數值欄位都已被用於維度拆解，無法選擇KPI。")
        st.stop()

    target_col = st.selectbox("請選擇要分析的 KPI (數值欄位)", available_kpis, key="page2_target_col")

    if target_col != st.session_state.get("page2_last_target_col") and st.session_state.get("page2_last_target_col") is not None:
        with st.spinner(f"KPI變更為 {target_col}，正在重算整棵樹..."):
            recalculated_layers = []
            for layer in st.session_state.get("rca_layers", []):
                df_sub = apply_filters_to_df(df_clean, layer["parent_filters"])
                new_groups = compute_top_groups_with_other(df_sub, target_col, layer["split_dim"], top_n=5)
                if new_groups:
                    recalculated_layers.append({**layer, "groups": new_groups})
            st.session_state["rca_layers"] = recalculated_layers
        st.toast(f"已使用新的KPI '{target_col}' 重算根因樹！")
    st.session_state["page2_last_target_col"] = target_col

    all_dims = [c for c in df_clean.columns if c != target_col]
    main_col, side_col = st.columns([4, 1.6], gap="large")

    with side_col:
        st.markdown("<div style='font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;border-bottom:1px solid #ccc;'>展開 / 維護樹</div>", unsafe_allow_html=True)
        
        parent_choices = build_parent_candidates_from_tail()
        parent_labels = [lab for lab, _ in parent_choices]
        sel_parent_label = st.selectbox("我要在哪個節點下面新增下一層？", options=parent_labels, key="rca_parent_select")
        parent_filters = next((fdict for lab, fdict in parent_choices if lab == sel_parent_label), {})

        df_parent_sub = apply_filters_to_df(df_clean, parent_filters)
        used_dims_here = {lyr["split_dim"] for lyr in st.session_state.get("rca_layers", []) if lyr["parent_filters"] == parent_filters}
        
        ranked_dims = rank_candidate_dims(df_parent_sub, target_col, all_dims, used_dims_here)

        if not ranked_dims:
            st.info("在這個節點下，沒有更多可拆的欄位。")
        else:
            sel_split_dim = st.selectbox("下一層要用哪個欄位分解？", options=ranked_dims, key="rca_split_dim_select")
            if st.button("➕ 加到樹裡 (展開下一層)"):
                add_new_layer(df_clean, target_col, parent_filters, sel_split_dim)
                st.rerun()

        st.markdown("<div style='font-size:0.9rem;font-weight:600;margin:1rem 0 0.5rem;border-bottom:1px solid #ccc;'>刪除某層拆解</div>", unsafe_allow_html=True)
        
        if st.session_state.get("rca_layers"):
            layer_labels = [f"{i}. 父[{('全體' if not lyr['parent_filters'] else ', '.join(lyr['parent_filters'].values()))}] → 拆分:[{lyr['split_dim']}]" for i, lyr in enumerate(st.session_state["rca_layers"])]
            
            sel_layer_to_remove_label = st.selectbox("選擇要刪除的層 (將移除此層及其後的所有層)：", options=layer_labels, index=len(layer_labels) - 1, key="rca_remove_layer_select")
            
            if st.button("🗑 刪除此層"):
                idx_to_remove = layer_labels.index(sel_layer_to_remove_label)
                remove_layer_by_index(idx_to_remove)
                st.rerun()

        st.markdown("<div style='font-size:0.9rem;font-weight:600;margin:1rem 0 0.5rem;border-bottom:1px solid #ccc;'>全部清除</div>", unsafe_allow_html=True)
        if st.button("💣 清空整棵樹 (回到只有KPI)"):
            st.session_state["rca_layers"] = []
            st.rerun()

    with main_col:
        st.markdown("<div style='font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;'>目前的根因樹</div>", unsafe_allow_html=True)
        html_tree = render_rca_tree_vertical(df_clean, target_col)
        st.markdown(html_tree, unsafe_allow_html=True)
        st.caption("說明：卡片顯示群組名稱、KPI 合計、以及對上層父節點的占比。藍色外框表示目前的分析路徑。")

########################################################
# 6. 頂部導覽列
########################################################
def render_top_nav():
    c1, c2 = st.columns([0.12, 0.88])
    with c1:
        st.markdown("<div style='font-size:0.7rem; color:#666; line-height:2.5;'>功能選單</div>", unsafe_allow_html=True)
    with c2:
        return st.radio("功能選單", ["📊 Excel表格分析", "🌳 根因分析樹"], horizontal=True, key="main_page_selector", label_visibility="collapsed")

########################################################
# 7. 主流程
########################################################
if "main_page_selector" not in st.session_state:
    st.session_state["main_page_selector"] = "📊 Excel表格分析"

current_page = render_top_nav()
st.markdown("<hr style='margin-top: -8px; margin-bottom: 8px;'>", unsafe_allow_html=True)

if current_page == "📊 Excel表格分析":
    page_table_analysis()
else:
    page_root_cause_tree()
