"""
Sales Data Analysis Dashboard
Navyan Data Analytics Internship - Project 1
Author : Kashish Tomar
Email  : kashishtomar36@gmail.com
GitHub : https://github.com/kashish0101
Dataset: Superstore Sales 2015-2018 | 9,800 rows | 19 columns
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "superstore.csv"
DATE_COLUMN = "Order Date"
OUTPUT_JSON = "dashboard_data.json"

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading data from: {path}")
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    print(f"    → {len(df):,} rows × {len(df.columns)} columns loaded")
    return df


# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Cleaning data ...")
    original_len = len(df)

    # Parse dates
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with no Sales value
    df = df.dropna(subset=["Sales"])

    # Remove negative or zero sales (invalid records)
    df = df[df["Sales"] > 0]

    # Strip whitespace from text columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Drop the empty Profit Margin column
    if "Profit Margin" in df.columns:
        df = df.drop(columns=["Profit Margin"])

    # Derive useful time columns
    df["Year"]      = df[DATE_COLUMN].dt.year.astype(int)
    df["Month"]     = df[DATE_COLUMN].dt.month
    df["YearMonth"] = df[DATE_COLUMN].dt.to_period("M")
    df["Quarter"]   = df[DATE_COLUMN].dt.to_period("Q").astype(str)

    removed = original_len - len(df)
    print(f"    → {removed:,} invalid rows removed. {len(df):,} clean rows remain.")
    return df


# ─────────────────────────────────────────────
# STEP 3: ANALYSE KEY METRICS
# ─────────────────────────────────────────────
def analyse(df: pd.DataFrame) -> dict:
    print("[3/5] Analysing key metrics ...")
    results = {}

    # ── KPIs ──────────────────────────────────
    results["kpi"] = {
        "total_sales"    : round(df["Sales"].sum(), 2),
        "total_orders"   : df["Order ID"].nunique(),
        "total_customers": df["Customer ID"].nunique(),
        "total_products" : df["Product Name"].nunique(),
        "avg_order_value": round(df["Sales"].mean(), 2),
    }

    # ── Monthly Sales Trend ───────────────────
    monthly = (
        df.groupby("YearMonth")["Sales"]
        .sum().reset_index().sort_values("YearMonth")
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    results["monthly_sales"] = monthly.to_dict(orient="records")

    # ── Yearly Sales ──────────────────────────
    yearly = df.groupby("Year")["Sales"].sum().reset_index()
    yearly["Growth_%"] = round(yearly["Sales"].pct_change() * 100, 1)
    results["yearly"] = yearly.to_dict(orient="records")

    # ── Sales by Region ───────────────────────
    region = df.groupby("Region")["Sales"].sum().reset_index()
    region = region.sort_values("Sales", ascending=False)
    results["region"] = region.to_dict(orient="records")

    # ── Sales by Category ─────────────────────
    category = df.groupby("Category")["Sales"].sum().reset_index()
    results["category"] = category.to_dict(orient="records")

    # ── Sales by Segment ──────────────────────
    segment = df.groupby("Segment")["Sales"].sum().reset_index()
    results["segment"] = segment.to_dict(orient="records")

    # ── Sales by Sub-Category (Top 10) ────────
    subcat = (
        df.groupby("Sub-Category")["Sales"]
        .sum().nlargest(10).reset_index()
        .sort_values("Sales")
    )
    results["subcategory"] = subcat.to_dict(orient="records")

    # ── Top 10 Products ───────────────────────
    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum().nlargest(10).reset_index()
        .sort_values("Sales")
    )
    results["top_products"] = top_products.to_dict(orient="records")

    # ── Monthly Seasonality (avg sales per month across years) ──
    seasonality = df.groupby("Month")["Sales"].mean().reset_index()
    seasonality.columns = ["Month", "Avg_Sales"]
    results["seasonality"] = seasonality.to_dict(orient="records")

    print("    → Analysis complete.")
    return results


# ─────────────────────────────────────────────
# STEP 4: VISUALISE
# ─────────────────────────────────────────────
PALETTE  = ["#2D6A4F", "#40916C", "#74C69D", "#52B788", "#B7E4C7"]
BG_COLOR = "#F8F9FA"
ACCENT   = "#2D6A4F"
MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"]

def _fmt_k(x, _):
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:     return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

def visualise(df: pd.DataFrame, results: dict):
    print("[4/5] Generating visualisations ...")
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig = plt.figure(figsize=(22, 18), facecolor=BG_COLOR)
    fig.suptitle(
        "Sales Performance Dashboard  —  Kashish Tomar\n"
        "Navyan Data Analytics Internship  |  Superstore Dataset 2015–2018",
        fontsize=18, fontweight="bold", color="#1B2631", y=0.98
    )
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.38)

    # ── 1. Monthly Sales Trend (full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    monthly = pd.DataFrame(results["monthly_sales"])
    ax1.plot(monthly["YearMonth"], monthly["Sales"],
             color=ACCENT, linewidth=2.2, marker="o", markersize=3)
    ax1.fill_between(monthly["YearMonth"], monthly["Sales"],
                     alpha=0.12, color=ACCENT)
    # Annotate peak month
    peak_idx = monthly["Sales"].idxmax()
    ax1.annotate(
        f"Peak: {_fmt_k(monthly['Sales'][peak_idx], None)}",
        xy=(monthly["YearMonth"][peak_idx], monthly["Sales"][peak_idx]),
        xytext=(0, 12), textcoords="offset points",
        ha="center", fontsize=8, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2)
    )
    ax1.set_title("Monthly Sales Trend (2015–2018)", fontweight="bold", pad=10)
    ax1.set_xlabel("")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ticks = monthly["YearMonth"].tolist()
    ax1.set_xticks(ticks[::3])
    ax1.set_xticklabels(ticks[::3], rotation=45, ha="right", fontsize=8)
    ax1.set_facecolor(BG_COLOR)

    # ── 2. Sales by Region ───────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    reg = pd.DataFrame(results["region"])
    bars = ax2.barh(reg["Region"], reg["Sales"],
                    color=PALETTE[:len(reg)], height=0.5)
    ax2.set_title("Sales by Region", fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_facecolor(BG_COLOR)
    for bar, val in zip(bars, reg["Sales"]):
        ax2.text(bar.get_width() * 1.01,
                 bar.get_y() + bar.get_height() / 2,
                 _fmt_k(val, None), va="center", fontsize=8, fontweight="bold")

    # ── 3. Sales by Category (donut) ─────────
    ax3 = fig.add_subplot(gs[1, 1])
    cat = pd.DataFrame(results["category"])
    total = cat["Sales"].sum()
    wedges, texts, autotexts = ax3.pie(
        cat["Sales"], labels=cat["Category"],
        colors=PALETTE[:len(cat)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"width": 0.55},
        textprops={"fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax3.set_title("Sales by Category", fontweight="bold")

    # ── 4. Sales by Segment ──────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    seg = pd.DataFrame(results["segment"]).sort_values("Sales", ascending=False)
    bars4 = ax4.bar(seg["Segment"], seg["Sales"],
                    color=PALETTE[:len(seg)], width=0.45)
    ax4.set_title("Sales by Segment", fontweight="bold")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax4.set_facecolor(BG_COLOR)
    for bar, val in zip(bars4, seg["Sales"]):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 _fmt_k(val, None), ha="center", fontsize=8, fontweight="bold")
    plt.setp(ax4.get_xticklabels(), rotation=15, ha="right")

    # ── 5. Top 10 Sub-Categories ─────────────
    ax5 = fig.add_subplot(gs[2, :2])
    sub = pd.DataFrame(results["subcategory"])
    colors5 = [PALETTE[i % len(PALETTE)] for i in range(len(sub))]
    bars5 = ax5.barh(sub["Sub-Category"], sub["Sales"],
                     color=colors5, height=0.6)
    ax5.set_title("Top 10 Sub-Categories by Sales", fontweight="bold")
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax5.set_facecolor(BG_COLOR)
    ax5.tick_params(axis="y", labelsize=9)
    for bar, val in zip(bars5, sub["Sales"]):
        ax5.text(bar.get_width() * 1.005,
                 bar.get_y() + bar.get_height() / 2,
                 _fmt_k(val, None), va="center", fontsize=8)

    # ── 6. Yearly Sales with growth labels ───
    ax6 = fig.add_subplot(gs[2, 2])
    yearly = pd.DataFrame(results["yearly"])
    bars6 = ax6.bar(yearly["Year"].astype(str), yearly["Sales"],
                    color=PALETTE[:len(yearly)], width=0.5)
    ax6.set_title("Yearly Sales (with Growth %)", fontweight="bold")
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax6.set_facecolor(BG_COLOR)
    for i, (bar, row) in enumerate(zip(bars6, yearly.itertuples())):
        label = _fmt_k(row.Sales, None)
        if i > 0 and not pd.isna(row._3):
            label += f"\n({row._3:+.1f}%)"
        ax6.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 label, ha="center", fontsize=7.5, fontweight="bold")

    plt.savefig("sales_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()
    print("    → Chart saved as: sales_dashboard.png")


# ─────────────────────────────────────────────
# STEP 5: EXPORT JSON FOR HTML DASHBOARD
# ─────────────────────────────────────────────
def export_json(results: dict):
    print(f"[5/5] Exporting data → {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("    → Export complete.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df  = load_data(DATA_PATH)
    df  = clean_data(df)
    res = analyse(df)

    kpi = res["kpi"]
    print("\n" + "═" * 46)
    print("   KEY PERFORMANCE INDICATORS")
    print("═" * 46)
    print(f"   Total Sales       : ${kpi['total_sales']:>13,.2f}")
    print(f"   Total Orders      : {kpi['total_orders']:>14,}")
    print(f"   Total Customers   : {kpi['total_customers']:>14,}")
    print(f"   Unique Products   : {kpi['total_products']:>14,}")
    print(f"   Avg Order Value   : ${kpi['avg_order_value']:>13,.2f}")
    print("═" * 46 + "\n")

    visualise(df, res)
    export_json(res)
    print("\n✅ All done! Open sales_dashboard.html in your browser.")

if __name__ == "__main__":
    main()
