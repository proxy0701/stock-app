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
    # 有効データが全銘柄の過半数未満の末尾行を除去
    # （yfinanceの日本株データ遅延で最終行がほぼNaNになるケースへの対策）
    valid_ratio = all_prices.notna().mean(axis=1)
    last_valid_idx = valid_ratio[valid_ratio >= 0.5].last_valid_index()
    if last_valid_idx is not None:
        all_prices = all_prices.loc[:last_valid_idx]
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


def compute_stock_rankings(sector: str, prices_df: pd.DataFrame, stocks_df: pd.DataFrame) -> pd.DataFrame:
    """指定業種に含まれる全銘柄の騰落率（全期間）を個別に計算する。"""
    group = stocks_df[stocks_df["33業種区分"] == sector]
    if group.empty:
        return pd.DataFrame()

    rows = []
    for _, stock in group.iterrows():
        ticker = stock["ticker"]
        if ticker not in prices_df.columns:
            continue

        series = prices_df[ticker].dropna()
        if series.empty:
            continue

        last_date = series.index[-1]
        last_close = series.iloc[-1]
        date_str = f"{last_date.month}/{last_date.day}"

        # 全データ（dropna前）のインデックスでの位置を取得
        full_series = prices_df[ticker]
        last_pos = full_series.index.get_loc(last_date)

        period_results = {}
        for period_name, n_days in PERIODS.items():
            past_pos = last_pos - n_days
            if past_pos < 0:
                period_results[period_name] = np.nan
                continue
            past_close = full_series.iloc[past_pos]
            if pd.isna(past_close) or past_close <= 0:
                period_results[period_name] = np.nan
                continue
            period_results[period_name] = float((last_close - past_close) / past_close * 100)

        rows.append({
            "コード": stock["コード"],
            "銘柄名": stock["銘柄名"],
            "日付": date_str,
            **period_results,
        })

    return pd.DataFrame(rows)


# ── テーブル描画 ────────────────────────────────────────────────────────────────

def _pct_cell(val, is_selected: bool) -> str:
    """騰落率の HTML セルを生成する（日本式: 上昇=赤、下落=青）。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<td style="text-align:right;color:#bbb;padding:6px 10px">--</td>'

    if val > 0:
        color = "#e63329"
        text = f"+{val:.2f}%"
    elif val < 0:
        color = "#1a73e8"
        text = f"{val:.2f}%"
    else:
        color = "#888"
        text = f"{val:.2f}%"

    style = f"text-align:right;color:{color};padding:6px 10px;"
    if is_selected:
        style += "font-weight:bold;"
        bg = "#fff3f3" if val > 0 else ("#f0f4ff" if val < 0 else "#f8f8f8")
        style += f"background:{bg};"

    return f'<td style="{style}">{text}</td>'


def render_ranking_table(df: pd.DataFrame, sort_col: str) -> None:
    """業種ランキングテーブルを HTML で描画する。"""
    df = df.copy()
    df["_sort"] = df[sort_col].apply(lambda x: x if pd.notna(x) else -9999.0)
    df = (
        df.sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    period_cols = list(PERIODS.keys())
    period_cols = [sort_col] + [p for p in period_cols if p != sort_col]
    th = "padding:8px 10px;white-space:nowrap;font-size:0.8rem;font-weight:600;color:#374151;"

    # ヘッダー行（選択中の列を強調表示）
    header_cells = (
        f'<th style="{th}text-align:center;width:2.8em">順位</th>'
        f'<th style="{th}text-align:left">業種名</th>'
        f'<th style="{th}text-align:right">銘柄数</th>'
    )
    for p in period_cols:
        if p == sort_col:
            header_cells += (
                f'<th style="{th}text-align:right;color:#b45309;'
                f'border-bottom:2px solid #d97706">{p} ▼</th>'
            )
        else:
            header_cells += f'<th style="{th}text-align:right">{p}</th>'

    # データ行
    rows_html = []
    for i, row in df.iterrows():
        rank = i + 1
        td = "padding:6px 10px;"

        cells = (
            f'<td style="{td}text-align:center;color:#9ca3af;font-size:0.82rem">{rank}</td>'
            f'<td style="{td}white-space:nowrap">'
            f'<a href="?sector={row["業種"]}" target="_self" '
            f'style="color:inherit;text-decoration:none;border-bottom:1px dashed #aaa">'
            f'{row["業種"]}</a></td>'
            f'<td style="{td}text-align:right;font-size:0.78rem;color:#9ca3af">'
            f'{int(row["銘柄数"])}社</td>'
        )
        for p in period_cols:
            cells += _pct_cell(row[p], is_selected=(p == sort_col))

        rows_html.append(f'<tr class="hover">{cells}</tr>')

    table_html = f"""
<div data-theme="corporate" class="fade-in"
     style="margin-top:0.5rem;overflow-x:auto;-webkit-overflow-scrolling:touch;
            border-radius:0.75rem;border:1px solid #e5e7eb;
            box-shadow:0 1px 4px rgba(0,0,0,0.06)">
  <table class="table table-zebra table-sm" style="min-width:600px;font-size:0.88rem">
    <thead style="background:#f8fafc">
      <tr>{header_cells}</tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def render_stock_table(df: pd.DataFrame, sort_col: str) -> None:
    """銘柄ランキングテーブルを HTML で描画する。"""
    df = df.copy()
    df["_sort"] = df[sort_col].apply(lambda x: x if pd.notna(x) else -9999.0)
    df = (
        df.sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    period_cols = list(PERIODS.keys())
    period_cols = [sort_col] + [p for p in period_cols if p != sort_col]
    th = "padding:8px 10px;white-space:nowrap;font-size:0.8rem;font-weight:600;color:#374151;"

    # ヘッダー行
    header_cells = (
        f'<th style="{th}text-align:center;width:2.8em">順位</th>'
        f'<th style="{th}text-align:left">銘柄名</th>'
        f'<th style="{th}text-align:right;color:#9ca3af">日付</th>'
    )
    for p in period_cols:
        if p == sort_col:
            header_cells += (
                f'<th style="{th}text-align:right;color:#b45309;'
                f'border-bottom:2px solid #d97706">{p} ▼</th>'
            )
        else:
            header_cells += f'<th style="{th}text-align:right">{p}</th>'

    # データ行
    rows_html = []
    for i, row in df.iterrows():
        rank = i + 1
        td = "padding:6px 10px;"

        cells = (
            f'<td style="{td}text-align:center;color:#9ca3af;font-size:0.82rem">{rank}</td>'
            f'<td style="{td}white-space:nowrap;font-weight:500">{row["銘柄名"]}</td>'
            f'<td style="{td}text-align:right;font-size:0.78rem;color:#9ca3af">{row["日付"]}</td>'
        )
        for p in period_cols:
            cells += _pct_cell(row[p], is_selected=(p == sort_col))

        rows_html.append(f'<tr class="hover">{cells}</tr>')

    table_html = f"""
<div data-theme="corporate" class="fade-in"
     style="margin-top:0.5rem;overflow-x:auto;-webkit-overflow-scrolling:touch;
            border-radius:0.75rem;border:1px solid #e5e7eb;
            box-shadow:0 1px 4px rgba(0,0,0,0.06)">
  <table class="table table-zebra table-sm" style="min-width:600px;font-size:0.88rem">
    <thead style="background:#f8fafc">
      <tr>{header_cells}</tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


@st.fragment
def show_ranking_section(rankings: pd.DataFrame) -> None:
    """期間選択 + ランキングテーブルをフラグメントとして描画する（期間切替時にここだけ再実行）。"""
    period_keys = list(PERIODS.keys())
    selected_period = st.segmented_control(
        "ソート期間",
        options=period_keys,
        default=period_keys[0],
        label_visibility="collapsed",
    )
    render_ranking_table(rankings, selected_period or period_keys[0])


@st.fragment
def show_stock_section(sector: str, prices_df: pd.DataFrame, stocks_df: pd.DataFrame) -> None:
    """銘柄ランキングセクションをフラグメントとして描画する（期間切替時にここだけ再実行）。"""
    st.markdown(
        '<a href="?" target="_self" '
        'style="font-size:0.9rem;color:#555;text-decoration:none">'
        '← 業種一覧に戻る</a>',
        unsafe_allow_html=True,
    )
    st.subheader(f"📋 {sector} — 銘柄別パフォーマンス")

    stock_rankings = compute_stock_rankings(sector, prices_df, stocks_df)
    if stock_rankings.empty:
        st.warning("この業種に該当する銘柄の価格データがありません。")
        return

    period_keys = list(PERIODS.keys())
    selected_period = st.segmented_control(
        "ソート期間",
        options=period_keys,
        default=period_keys[0],
        label_visibility="collapsed",
    )
    render_stock_table(stock_rankings, selected_period or period_keys[0])


# ── メイン ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="東証プライム 33業種パフォーマンス",
        page_icon="📊",
        layout="centered",
    )

    st.markdown(
        """
<link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet" type="text/css" />
<style>
.block-container{padding-top:1.5rem;padding-bottom:1rem}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeIn 0.35s ease-out}
html,body,[data-testid="stAppViewContainer"],.main{background-color:#f5f7fa !important}
/* Streamlitデフォルトタイトルを非表示にして独自ヘッダーを使う */
[data-testid="stAppViewContainer"] > section > div > div:first-child h1{display:none}
</style>
<div data-theme="corporate" class="fade-in"
     style="background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 100%);
            border-radius:1rem;padding:1.25rem 1.5rem;color:#fff;
            margin-top:2.5rem;margin-bottom:0.25rem">
  <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.3px">
    📊 東証プライム 33業種パフォーマンス
  </div>
  <div style="font-size:0.82rem;opacity:0.8;margin-top:0.3rem">
    プライム市場 全33業種の騰落率ランキング（時価総額加重平均）
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 銘柄一覧の読み込み
    with st.spinner("銘柄一覧を読み込み中..."):
        stocks_df = load_stock_list()

    if stocks_df.empty:
        st.warning("銘柄データが読み込めませんでした。CSVファイルを確認してください。")
        return

    today = datetime.today().strftime("%Y-%m-%d")
    st.markdown(
        f'<div data-theme="corporate" style="margin:0.4rem 0 0.8rem;font-size:0.78rem;color:#6b7280">'
        f'対象: プライム市場 <strong>{len(stocks_df):,}社</strong>'
        f' &nbsp;|&nbsp; 更新: {today}（日次キャッシュ）</div>',
        unsafe_allow_html=True,
    )

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

    # ランキング表示（フラグメント：期間切替時にここだけ再実行）
    sector = st.query_params.get("sector")
    if sector and sector in TSE_33_SECTORS:
        show_stock_section(sector, prices_df, stocks_df)
    else:
        show_ranking_section(rankings)

    st.markdown(
        '<div data-theme="corporate" style="margin-top:1.5rem;padding:0.75rem 1rem;'
        'background:#f8fafc;border-radius:0.5rem;border:1px solid #e5e7eb;'
        'font-size:0.75rem;color:#9ca3af;line-height:1.6">'
        '※ 時価総額加重平均方式（最新株価 × 発行済株式数）で計算。'
        '発行済株式数はyfinanceから自動取得（日次更新）。'
        '騰落率は最新取得終値ベース（前営業日終値）。'
        '一部銘柄はデータ取得できない場合があります。'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
