import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "tse_stocks.csv")
PRIME_MARKET = "プライム（内国株式）"
PERIODS = {"1日": 1, "3日": 3, "1週間": 5, "1ヶ月": 21, "3ヶ月": 63, "6ヶ月": 118}
BATCH_SIZE = 500

TSE_33_SECTORS = [
    "水産・農林業", "鉱業", "建設業", "食料品", "繊維製品", "パルプ・紙",
    "化学", "医薬品", "石油・石炭製品", "ゴム製品", "ガラス・土石製品",
    "鉄鋼", "非鉄金属", "金属製品", "機械", "電気機器", "輸送用機器",
    "精密機器", "その他製品", "電気・ガス業", "陸運業", "海運業", "空運業",
    "倉庫・運輸関連業", "情報・通信業", "卸売業", "小売業", "銀行業",
    "証券、商品先物取引業", "保険業", "その他金融業", "不動産業", "サービス業",
]


# ── データ読み込み ──────────────────────────────────────────────────────────────

@st.cache_data(ttl="1d", show_spinner=False)
def load_stock_list() -> pd.DataFrame:
    """JPX公式「東証上場銘柄一覧」CSVを読み込み、プライム市場・普通株のみ返す。"""
    if not os.path.exists(CSV_PATH):
        st.error(
            "📁 `data/tse_stocks.csv` が見つかりません。\n\n"
            "JPX公式サイトから「東証上場銘柄一覧」をダウンロードし、"
            "`data/tse_stocks.csv` として保存してください。"
        )
        st.stop()

    df = None
    for enc in ("shift_jis", "cp932", "utf-8"):
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc, dtype={"コード": str}, thousands=",")
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        st.error("CSVの読み込みに失敗しました。ファイルのエンコーディングを確認してください。")
        st.stop()

    # プライム市場のみに絞る
    if "市場・商品区分" in df.columns:
        df = df[df["市場・商品区分"] == PRIME_MARKET].copy()

    # 33業種区分が空の行（ETF・REIT等）を除外
    if "33業種区分" not in df.columns:
        st.error("CSVに「33業種区分」列が見つかりません。JPX公式の列名をご確認ください。")
        st.stop()

    df = df[df["33業種区分"].notna() & (df["33業種区分"].astype(str).str.strip() != "")].copy()

    # ticker列生成（コードを4桁ゼロ埋め + ".T"）
    df["ticker"] = df["コード"].astype(str).str.strip().str.zfill(4) + ".T"

    # 発行済株式数の取得（列名は「発行済株式数（千株）」または「発行済株式数」）
    if "発行済株式数（千株）" in df.columns:
        df["shares"] = (
            pd.to_numeric(
                df["発行済株式数（千株）"].astype(str).str.replace(",", ""),
                errors="coerce",
            )
            * 1000
        )
    elif "発行済株式数" in df.columns:
        df["shares"] = pd.to_numeric(
            df["発行済株式数"].astype(str).str.replace(",", ""),
            errors="coerce",
        )
    else:
        # 列が存在しない場合は等加重で計算するため NaN のまま
        df["shares"] = np.nan

    return df[["コード", "銘柄名", "33業種区分", "ticker", "shares"]].reset_index(drop=True)


@st.cache_data(ttl="1d", show_spinner=False)
def fetch_shares(tickers: tuple) -> dict:
    """全銘柄の発行済株式数を yfinance から並列取得する（1日キャッシュ）。"""

    def _get_one(ticker: str) -> tuple[str, float | None]:
        try:
            fi = yf.Ticker(ticker).fast_info
            s = getattr(fi, "shares", None)
            if s and s > 0:
                return ticker, float(s)
            # shares が取れない場合は時価総額 ÷ 株価で推計
            mc = getattr(fi, "market_cap", None)
            lp = getattr(fi, "last_price", None)
            if mc and lp and lp > 0:
                return ticker, float(mc / lp)
        except Exception:
            pass
        return ticker, None

    result = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        for ticker, shares in executor.map(_get_one, tickers):
            if shares:
                result[ticker] = shares
    return result


@st.cache_data(ttl="1d", show_spinner=False)
def fetch_all_prices(tickers: tuple) -> pd.DataFrame:
    """全銘柄の過去6ヶ月分の終値を取得してDataFrameで返す。"""
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    results = []

    progress_bar = st.progress(0, text="株価データを取得中...")

    for i, batch in enumerate(batches):
        try:
            raw = yf.download(
                list(batch),
                period="6mo",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                continue

            # MultiIndex 対応（複数銘柄）vs 単一銘柄
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            results.append(close)

        except Exception:
            pass

        progress_bar.progress(
            (i + 1) / len(batches),
            text=f"株価データを取得中... {i + 1}/{len(batches)} バッチ完了",
        )

    progress_bar.empty()

    if not results:
        return pd.DataFrame()

    all_prices = pd.concat(results, axis=1)
    # 列名の重複を除去
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
    return all_prices


# ── パフォーマンス計算 ──────────────────────────────────────────────────────────

def compute_sector_rankings(prices_df: pd.DataFrame, stocks_df: pd.DataFrame) -> pd.DataFrame:
    """33業種ごとに時価総額加重平均の騰落率（全期間）を計算する。"""
    rows = []

    for sector in TSE_33_SECTORS:
        group = stocks_df[stocks_df["33業種区分"] == sector]
        if group.empty:
            rows.append({"業種": sector, "銘柄数": 0, **{p: np.nan for p in PERIODS}})
            continue

        # 価格データが存在する銘柄のみ抽出
        valid_tickers = [t for t in group["ticker"] if t in prices_df.columns]
        if not valid_tickers:
            rows.append({"業種": sector, "銘柄数": 0, **{p: np.nan for p in PERIODS}})
            continue

        sector_prices = prices_df[valid_tickers].dropna(axis=1, how="all")
        if sector_prices.empty:
            rows.append({"業種": sector, "銘柄数": 0, **{p: np.nan for p in PERIODS}})
            continue

        latest_close = sector_prices.iloc[-1]

        # 時価総額ウェイト計算（発行済株式数 × 最新株価）
        shares_map = group.set_index("ticker")["shares"]
        shares = shares_map.reindex(sector_prices.columns)

        if shares.isna().all():
            # 発行済株式数データなし → 等加重
            base_weights = pd.Series(1.0, index=sector_prices.columns)
        else:
            base_weights = latest_close * shares.fillna(0)

        # 各期間の加重平均騰落率を計算
        period_results = {}
        for period_name, n_days in PERIODS.items():
            idx = len(sector_prices) - 1 - n_days
            if idx < 0:
                period_results[period_name] = np.nan
                continue

            past_close = sector_prices.iloc[idx]
            valid_mask = (past_close > 0) & past_close.notna() & latest_close.notna()

            if not valid_mask.any():
                period_results[period_name] = np.nan
                continue

            change_rate = (latest_close[valid_mask] - past_close[valid_mask]) / past_close[valid_mask]
            weights = base_weights[valid_mask]
            total_weight = weights.sum()

            if total_weight == 0:
                period_results[period_name] = np.nan
            else:
                period_results[period_name] = float((change_rate * weights).sum() / total_weight * 100)

        rows.append({
            "業種": sector,
            "銘柄数": len(sector_prices.columns),
            **period_results,
        })

    return pd.DataFrame(rows)


# ── テーブル描画 ────────────────────────────────────────────────────────────────

def _pct_cell(val, is_selected: bool) -> str:
    """騰落率の HTML セルを生成する（日本式: 上昇=赤、下落=青）。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<td style="text-align:right;color:#bbb;padding:6px 8px">--</td>'

    if val > 0:
        color = "#e63329"
        text = f"+{val:.2f}%"
    elif val < 0:
        color = "#1a73e8"
        text = f"{val:.2f}%"
    else:
        color = "#888"
        text = f"{val:.2f}%"

    style = f"text-align:right;color:{color};padding:6px 8px;"
    if is_selected:
        style += "font-weight:bold;"
        bg = "#fff3f3" if val > 0 else ("#f0f4ff" if val < 0 else "#f8f8f8")
        style += f"background:{bg};"

    return f'<td style="{style}">{text}</td>'


def render_ranking_table(df: pd.DataFrame, sort_col: str) -> None:
    """業種ランキングテーブルを HTML で描画する。"""
    from urllib.parse import quote

    df = df.copy()
    df["_sort"] = df[sort_col].apply(lambda x: x if pd.notna(x) else -9999.0)
    df = (
        df.sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    period_cols = list(PERIODS.keys())
    th = "padding:6px 8px;white-space:nowrap;"

    # ヘッダー行（期間列はクリックでソート切替）
    header_cells = (
        f'<th style="{th}text-align:center;width:3em">順位</th>'
        f'<th style="{th}text-align:left">業種名</th>'
        f'<th style="{th}text-align:right">銘柄数</th>'
    )
    for p in period_cols:
        if p == sort_col:
            sel_style = "font-weight:bold;border-bottom:3px solid #e63329;color:#e63329;"
            header_cells += f'<th style="{th}text-align:right;{sel_style}">{p}</th>'
        else:
            header_cells += (
                f'<th style="{th}text-align:right;">'
                f'<a href="?sort={quote(p)}" target="_top" style="color:inherit;text-decoration:none;cursor:pointer;">{p}</a>'
                f'</th>'
            )

    # データ行
    rows_html = []
    for i, row in df.iterrows():
        rank = i + 1
        bg = "#fafafa" if rank % 2 == 0 else "#ffffff"
        td = "padding:6px 8px;"

        cells = (
            f'<td style="{td}text-align:center;color:#999;font-size:0.85em">{rank}</td>'
            f'<td style="{td}white-space:nowrap">{row["業種"]}</td>'
            f'<td style="{td}text-align:right;color:#aaa;font-size:0.8em">'
            f'{int(row["銘柄数"])}社</td>'
        )
        for p in period_cols:
            cells += _pct_cell(row[p], is_selected=(p == sort_col))

        rows_html.append(f'<tr style="background:{bg}">{cells}</tr>')

    table_html = f"""
<div style="overflow-x:auto;margin-top:0.5rem">
<table style="width:100%;border-collapse:collapse;font-size:0.88rem">
  <thead>
    <tr style="border-bottom:2px solid #ddd">{header_cells}</tr>
  </thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


# ── メイン ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="東証プライム 33業種パフォーマンス",
        page_icon="📊",
        layout="centered",
    )

    st.markdown(
        "<style>.block-container{padding-top:1.2rem;padding-bottom:1rem}</style>",
        unsafe_allow_html=True,
    )

    st.title("📊 東証プライム 33業種パフォーマンス")

    # 銘柄一覧の読み込み
    with st.spinner("銘柄一覧を読み込み中..."):
        stocks_df = load_stock_list()

    if stocks_df.empty:
        st.warning("銘柄データが読み込めませんでした。CSVファイルを確認してください。")
        return

    # ソート列を URL パラメータから取得（テーブルヘッダークリックで切替）
    today = datetime.today().strftime("%Y-%m-%d")
    selected_period = st.query_params.get("sort", "1日")
    if selected_period not in PERIODS:
        selected_period = "1日"

    st.caption(f"対象: プライム市場 {len(stocks_df):,}社 ｜ ソート: {selected_period} ｜ 更新: {today}（日次キャッシュ）")

    tickers = tuple(stocks_df["ticker"].tolist())

    # 発行済株式数の取得（CSVに列がない場合は yfinance から自動取得）
    if stocks_df["shares"].isna().all():
        with st.spinner("発行済株式数を取得中（初回のみ・約1〜2分）..."):
            shares_dict = fetch_shares(tickers)
        stocks_df = stocks_df.copy()
        stocks_df["shares"] = stocks_df["ticker"].map(shares_dict)

    # 株価データ取得
    with st.spinner("株価データを準備中（初回は数分かかる場合があります）..."):
        prices_df = fetch_all_prices(tickers)

    if prices_df.empty:
        st.error("株価データを取得できませんでした。しばらく待ってから再度お試しください。")
        return

    # 業種別パフォーマンス計算
    with st.spinner("業種別パフォーマンスを計算中..."):
        rankings = compute_sector_rankings(prices_df, stocks_df)

    if rankings.empty:
        st.warning("ランキングデータを計算できませんでした。")
        return

    # ランキング表示
    render_ranking_table(rankings, selected_period)

    st.markdown("---")
    st.caption(
        "※ 時価総額加重平均方式（最新株価 × 発行済株式数）で計算。"
        "発行済株式数はyfinanceから自動取得（日次更新）。"
        "騰落率は最新取得終値ベース（前営業日終値）。"
        "一部銘柄はデータ取得できない場合があります。"
    )


if __name__ == "__main__":
    main()
