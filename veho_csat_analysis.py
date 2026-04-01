# =============================================================================
# VEHO CSAT ANALYSIS
# Exploratory analysis of customer satisfaction data from Veho delivery
# operations. This script loads raw case study data, inspects its structure,
# and produces a series of charts to understand CSAT drivers across markets,
# events, and time periods.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
import os
import datetime

# =============================================================================
# CHART THEME
# Apply a dark theme globally so every figure inherits consistent styling
# without needing per-chart configuration. Using rcParams ensures third-party
# helpers (seaborn, pandas .plot()) also pick up the theme.
# =============================================================================

plt.rcParams.update({
    "figure.facecolor":  "#0c0f14",
    "axes.facecolor":    "#151920",
    "savefig.facecolor": "#0c0f14",
    "text.color":        "#e8eaed",
    "axes.labelcolor":   "#e8eaed",
    "xtick.color":       "#e8eaed",
    "ytick.color":       "#e8eaed",
    "axes.edgecolor":    "#2a2f3a",
    "grid.color":        "#2a2f3a",
    "grid.linewidth":    0.6,
    "figure.figsize":    (12, 7),
    "figure.dpi":        150,
    "font.family":       "sans-serif",
})

# =============================================================================
# COLOR PALETTE
# Named constants keep chart code readable and make palette swaps trivial.
# =============================================================================

ACCENT     = "#ff6b35"
GREEN      = "#34d399"
RED        = "#f87171"
BLUE       = "#60a5fa"
YELLOW     = "#fbbf24"
PURPLE     = "#a78bfa"
TEXT_MUTED = "#8b8f98"

# =============================================================================
# OUTPUT DIRECTORY
# All figures are written to figures/ so they stay out of the repo root and
# are easy to attach to a slide deck or report.
# =============================================================================

os.makedirs("figures", exist_ok=True)


def save_fig(fig, name):
    """Save a figure to figures/<name>.png then display it inline.

    Using bbox_inches='tight' prevents axis labels from being clipped.
    The facecolor kwarg is required here too because savefig resets it
    unless explicitly passed.
    """
    path = os.path.join("figures", f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> saved {path}")
    plt.show()


# =============================================================================
# DATA LOADING
# Read the raw CSV exported from the Veho case study dataset. We deliberately
# avoid any dtype coercion here so we can inspect exactly what the raw values
# look like before deciding on a cleaning strategy.
# =============================================================================

CSV_PATH = "case_study_veho_Raw_Data.csv"
df = pd.read_csv(CSV_PATH, low_memory=False)

# =============================================================================
# INITIAL INSPECTION
# Print structural metadata so we can spot schema issues (wrong dtypes,
# unexpected nulls, column naming) before writing any transformation logic.
# =============================================================================

print("=" * 60)
print("SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DTYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("NULL COUNTS")
print("=" * 60)
print(df.isnull().sum())

# =============================================================================
# CATEGORICAL FIELD CARDINALITY
# market and last_event are the two primary segmentation dimensions. Printing
# their unique values up front reveals any inconsistent casing, trailing
# spaces, or unexpected categories before we build any groupby logic.
# =============================================================================

for col in ["market", "last_event"]:
    if col in df.columns:
        print(f"\n{'=' * 60}")
        print(f"UNIQUE VALUES -- {col}  ({df[col].nunique()} distinct)")
        print("=" * 60)
        print(df[col].value_counts(dropna=False).to_string())
    else:
        print(f"\n[WARNING] Column '{col}' not found in dataset.")

# =============================================================================
# CSAT FIELD SAMPLE
# The CSAT column is frequently messy in raw exports — it may contain numeric
# strings, emoji, free-text ratings like "5/5", nulls, or mixed scales.
# Printing 20 random raw values before any parsing gives us a clear picture
# of what cleaning logic we actually need.
# =============================================================================

csat_candidates = [c for c in df.columns if "csat" in c.lower()]
print(f"\n{'=' * 60}")
print(f"CSAT COLUMN CANDIDATES: {csat_candidates}")
print("=" * 60)

for col in csat_candidates:
    print(f"\n--- 20 sample values from '{col}' ---")
    sample = df[col].dropna().sample(min(20, df[col].dropna().shape[0]), random_state=42)
    for val in sample:
        print(f"  {repr(val)}")


# =============================================================================
# PHASE 2 — DATA CLEANING
# =============================================================================

# =============================================================================
# CSAT EXTRACTION
# The csat field is a free-text mess: drivers type ratings alongside comments,
# emojis, punctuation, and annotations. We need a single integer 1-5 or NaN.
#
# The extraction strategy uses a priority ladder so that the most unambiguous
# patterns are tried first. Falling through to "any digit 1-5 anywhere" last
# prevents false positives (e.g. extracting a "3" from a phone number fragment)
# while still recovering ratings buried inside longer strings.
# =============================================================================

def extract_csat(val):
    """Extract a 1-5 integer rating from a messy free-text CSAT string.

    Priority order (first match wins):
      1. Bare digit string          "5"  / "4"
      2. Digit with trailing +      "4+"
      3. String that starts with    "5 thank you!" → 5
         a 1-5 digit
      4. Digit in parentheses       "(4) almost great"
      5. Trailing digit             "Thank you! 5"
      6. Emoji-prefixed digit       first char is 1-5
      7. Any isolated digit 1-5     last resort scan
      8. NaN — could not parse
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()

    # 1. Bare digit — the clean, happy-path case
    if re.fullmatch(r"[1-5]", s):
        return int(s)

    # 2. Digit with trailing modifier like "4+"
    m = re.fullmatch(r"([1-5])\+", s)
    if m:
        return int(m.group(1))

    # 3. String that starts with a 1-5 digit followed by a non-digit boundary
    #    e.g. "5 thank you!", "5👍"
    m = re.match(r"^([1-5])(?:\D|$)", s)
    if m:
        return int(m.group(1))

    # 4. Digit wrapped in parentheses "(4) almost great 😊"
    m = re.search(r"\(([1-5])\)", s)
    if m:
        return int(m.group(1))

    # 5. Trailing digit at the end of the string "Thank you! 5"
    m = re.search(r"(?<!\d)([1-5])(?!\d)\s*$", s)
    if m:
        return int(m.group(1))

    # 6. First character is a digit 1-5 (handles emoji concatenation edge cases
    #    where regex word boundaries may not fire, e.g. "5👍" caught above,
    #    but belt-and-suspenders for other encodings)
    if s[0] in "12345":
        return int(s[0])

    # 7. Last resort: any isolated 1-5 digit anywhere in the string.
    #    "Actually, 1. It's not ours" → 1
    m = re.search(r"(?<!\d)([1-5])(?!\d)", s)
    if m:
        return int(m.group(1))

    return np.nan


def extract_comment(val):
    """Strip the leading numeric rating tokens and return leftover comment text.

    Removes patterns like "5", "4+", "(4)", "5 -", etc. from the front of the
    string. Returns the remaining text if it is meaningful (>3 chars after
    stripping), otherwise returns NaN so comment columns stay sparse/clean.
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()

    # Strip leading rating tokens: optional paren-wrapped digit, bare digit,
    # optional +, optional punctuation/dash separator
    s = re.sub(r"^\(?[1-5]\)?\+?\s*[-–—:]?\s*", "", s).strip()

    return s if len(s) > 3 else np.nan


# Apply both extractors. Using a dedicated column preserves the raw value for
# auditing and lets us diff the parser output against originals easily.
CSAT_COL = csat_candidates[0] if csat_candidates else "csat"

df["csat_numeric"] = df[CSAT_COL].apply(extract_csat)
df["csat_comment"] = df[CSAT_COL].apply(extract_comment)

# =============================================================================
# DATETIME PARSING
# route_start_time and updated_at arrive as ISO8601 strings. Parsing with
# utc=True normalises any mixed-offset data to a single UTC-aware dtype,
# which prevents silent errors when we later compute durations or group by
# local date.
# =============================================================================

for col in ["route_start_time", "updated_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True, errors="coerce")

# Engineer time-based features from updated_at (the timestamp closest to when
# the rating was actually submitted). delivery_duration_hrs measures how long
# the route ran, which may correlate with driver fatigue and low scores.
if "updated_at" in df.columns:
    df["delivery_date"]  = df["updated_at"].dt.date
    df["day_of_week"]    = df["updated_at"].dt.day_name()
    df["hour_of_day"]    = df["updated_at"].dt.hour
    df["week_number"]    = df["updated_at"].dt.isocalendar().week.astype(int)

if "route_start_time" in df.columns and "updated_at" in df.columns:
    df["delivery_duration_hrs"] = (
        (df["updated_at"] - df["route_start_time"])
        .dt.total_seconds()
        .div(3600)
        # Negative or astronomically large values are data errors; null them out
        .where(lambda x: x.between(0, 24))
    )

# =============================================================================
# GEO NORMALIZATION
# city and state arrive in inconsistent formats (mixed case, full state names).
# Uppercasing city is sufficient; state needs a lookup table because we want
# uniform 2-letter codes for groupby and labelling.
# =============================================================================

STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

def normalize_state(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Already a 2-letter code — just uppercase it
    if re.fullmatch(r"[A-Za-z]{2}", s):
        return s.upper()
    return STATE_MAP.get(s.lower(), s.upper())

if "city" in df.columns:
    df["city"] = df["city"].str.upper().str.strip()

if "state" in df.columns:
    df["state"] = df["state"].apply(normalize_state)

# =============================================================================
# VALID SUBSET
# All downstream analysis requires a numeric CSAT score. We create a filtered
# view rather than dropping rows in-place so the full raw df stays available
# for auditing unparsed records.
# =============================================================================

valid = df[df["csat_numeric"].notna()].copy()

# =============================================================================
# CLEANING SUMMARY
# Printing these stats up front makes it easy to spot if the parser is
# under-performing (low parse rate) or if date coverage is unexpectedly narrow.
# =============================================================================

total_rows    = len(df)
valid_count   = len(valid)
parse_rate    = valid_count / total_rows * 100 if total_rows else 0
comment_count = valid["csat_comment"].notna().sum()

date_min = valid["updated_at"].min() if "updated_at" in valid.columns else "N/A"
date_max = valid["updated_at"].max() if "updated_at" in valid.columns else "N/A"

# Collect a sample of raw values that the parser could not resolve so we can
# iterate on the regex if the failure mode is systematic
unparsed_sample = (
    df[df["csat_numeric"].isna()][CSAT_COL]
    .dropna()
    .sample(min(10, df[df["csat_numeric"].isna()][CSAT_COL].dropna().shape[0]), random_state=42)
    .tolist()
)

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"  Total rows          : {total_rows:,}")
print(f"  Valid CSAT rows     : {valid_count:,}")
print(f"  Parse rate          : {parse_rate:.1f}%")
print(f"  Date range          : {date_min}  ->  {date_max}")
print(f"  Rows with comments  : {comment_count:,}")
print(f"\n  Sample unparsed values ({len(unparsed_sample)} shown):")
for v in unparsed_sample:
    print(f"    {repr(v)}")


# =============================================================================
# PHASE 3 — CSAT DISTRIBUTION
# Two charts that expose the extreme right-skew of the rating distribution and
# make the 4.9 vs 5.0 gap visually legible to a non-technical audience.
# =============================================================================

from scipy import stats as scipy_stats

# ── Rating counts & derived percentages ─────────────────────────────────────
rating_counts = valid["csat_numeric"].value_counts().sort_index()
rating_pct    = rating_counts / rating_counts.sum() * 100

BAR_COLORS = {1: RED, 2: "#fb923c", 3: YELLOW, 4: BLUE, 5: GREEN}

# ── Descriptive stats (needed by both charts and the printed insight) ────────
csat_vals  = valid["csat_numeric"].dropna()
mean_val   = csat_vals.mean()
median_val = csat_vals.median()
std_val    = csat_vals.std()
skew_val   = scipy_stats.skew(csat_vals)
pct_5      = rating_pct.get(5, 0)
pct_1      = rating_pct.get(1, 0)

print("\n" + "=" * 60)
print("PHASE 3 -- CSAT DESCRIPTIVE STATS")
print("=" * 60)
print(f"  Mean     : {mean_val:.4f}")
print(f"  Median   : {median_val:.1f}")
print(f"  Std Dev  : {std_val:.4f}")
print(f"  Skewness : {skew_val:.4f}")
print()
for rating in [1, 2, 3, 4, 5]:
    cnt = int(rating_counts.get(rating, 0))
    pct = rating_pct.get(rating, 0.0)
    print(f"  {rating}*  {cnt:>6,}  ({pct:.2f}%)")
print()
print(
    f"  Insight: {pct_5:.1f}% of ratings are 5-star. "
    f"The gap to 4.9 is driven by {pct_1:.1f}% of 1-star ratings."
)

# =============================================================================
# CHART 1 — 01_csat_distribution.png
# Bar chart with LOG-scale y-axis. Each bar is coloured by star tier.
# Every bar is annotated with its count and percentage. A horizontal dashed
# reference line marks the height of the 5-star bar and is labelled with the
# computed mean so viewers can immediately see that 5-star dominance is what
# holds the overall average near 4.9.
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.bar(
    rating_counts.index,
    rating_counts.values,
    color=[BAR_COLORS[r] for r in rating_counts.index],
    edgecolor="#0c0f14",
    linewidth=0.8,
    width=0.62,
    zorder=3,
)

ax.set_yscale("log")
ax.set_xlabel("Star Rating", fontsize=13, labelpad=8)
ax.set_ylabel("Count (log scale)", fontsize=13, labelpad=8)
ax.set_title("CSAT Rating Distribution (Log Scale)", fontsize=15, fontweight="bold", pad=14)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1★", "2★", "3★", "4★", "5★"], fontsize=13)

# Annotate each bar: count on top, percentage just below it
for bar, (rating, cnt) in zip(bars, rating_counts.items()):
    pct = rating_pct[rating]
    y   = bar.get_height()
    # Offset chosen so text clears the bar on a log scale
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y * 1.6,
        f"{int(cnt):,}",
        ha="center", va="bottom",
        fontsize=10, color="#e8eaed", fontweight="bold",
        zorder=4,
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y * 1.15,
        f"({pct:.1f}%)",
        ha="center", va="bottom",
        fontsize=9, color=TEXT_MUTED,
        zorder=4,
    )

# Horizontal dashed line at the 5-star count as a visual reference for the
# dominant rating tier, annotated with the computed mean CSAT.
count_5 = int(rating_counts.get(5, 1))
ax.axhline(
    count_5,
    color=TEXT_MUTED,
    linewidth=1.2,
    linestyle="--",
    alpha=0.65,
    zorder=2,
)
ax.text(
    1.3,
    count_5 * 1.08,
    f"Avg CSAT ≈ {mean_val:.1f}",
    color=TEXT_MUTED,
    fontsize=10,
    va="bottom",
    zorder=4,
)

ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.45, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0.4, 5.6)

plt.tight_layout()
save_fig(fig, "01_csat_distribution")
plt.close(fig)

# =============================================================================
# CHART 2 — 02_csat_violin.png
# Violin plot (showing the full density shape) with a narrow box-and-whisker
# overlaid at the same x-position. Because the data is nearly all 5-star the
# violin will have an enormous mass at y=5 and thin tails downward — this
# makes the extreme left skew immediately visible in a way a bar chart cannot.
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

csat_arr = csat_vals.values

# Violin — matplotlib gives us full colour/alpha control without seaborn
vp = ax.violinplot(
    [csat_arr],
    positions=[1],
    widths=0.7,
    showmedians=False,
    showextrema=False,
)
for body in vp["bodies"]:
    body.set_facecolor(BLUE)
    body.set_edgecolor("#8ba8d4")
    body.set_alpha(0.45)
    body.set_linewidth(0.8)
    body.set_zorder(2)

# Box + whiskers overlaid on the violin
bp = ax.boxplot(
    [csat_arr],
    positions=[1],
    widths=0.09,
    patch_artist=True,
    zorder=3,
    boxprops=dict(facecolor=ACCENT, color="#e8eaed", linewidth=1.3),
    medianprops=dict(color=GREEN, linewidth=2.2),
    whiskerprops=dict(color="#e8eaed", linewidth=1.2, linestyle="--"),
    capprops=dict(color="#e8eaed", linewidth=1.4),
    flierprops=dict(
        marker="o",
        markerfacecolor=RED,
        markeredgecolor="none",
        markersize=4,
        alpha=0.35,
        linestyle="none",
    ),
)

ax.set_ylabel("CSAT Rating", fontsize=13, labelpad=8)
ax.set_title(
    "CSAT Rating Distribution — Violin + Box\n(Extreme Left Skew Visible)",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1★", "2★", "3★", "4★", "5★"], fontsize=12)
ax.set_xticks([])
ax.set_xlim(0.4, 1.6)
ax.set_ylim(0.3, 5.7)

# Stats annotation in bottom-right corner
stats_text = (
    f"Mean: {mean_val:.2f}   Median: {int(median_val)}"
    f"\nStd: {std_val:.3f}   Skewness: {skew_val:.3f}"
)
ax.text(
    0.97, 0.03,
    stats_text,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=10, color=TEXT_MUTED,
    linespacing=1.6,
)

# Per-rating % labels along the right edge for quick reading
for rating in [1, 2, 3, 4, 5]:
    pct = rating_pct.get(rating, 0.0)
    if pct > 0:
        ax.text(
            1.38,
            rating,
            f"{pct:.1f}%",
            va="center", ha="left",
            fontsize=10,
            color=BAR_COLORS[rating],
            fontweight="bold",
        )

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig(fig, "02_csat_violin")
plt.close(fig)


# =============================================================================
# PHASE 4 -- MARKET ANALYSIS
# Three charts that compare performance across Veho's five markets and a
# formatted summary table printed to stdout.
# =============================================================================

TARGET_CSAT = 4.9  # reference line used across market charts

# ── Market-level aggregation ─────────────────────────────────────────────────
# We need: mean CSAT, 5-star rate, below-4-star rate, and misdelivery rate.
# All delivery-outcome metrics use the full df (not just rows with ratings) so
# the denominators are correct for each market.

market_csat = (
    valid.groupby("market")["csat_numeric"]
    .agg(
        count="count",
        mean_csat="mean",
    )
    .assign(
        five_star_pct=lambda d: valid.groupby("market")["csat_numeric"]
            .apply(lambda s: (s == 5).sum() / len(s) * 100),
        below4_pct=lambda d: valid.groupby("market")["csat_numeric"]
            .apply(lambda s: (s < 4).sum() / len(s) * 100),
    )
    .reset_index()
)

# Misdelivery rate uses full df so markets with no CSAT ratings still count
misdelivery = (
    df.groupby("market")["last_event"]
    .agg(
        total="count",
        misdelivered=lambda s: (s == "misdelivered").sum(),
    )
    .assign(misdelivery_rate=lambda d: d["misdelivered"] / d["total"] * 100)
    .reset_index()
)

market_stats = market_csat.merge(misdelivery, on="market")
market_stats = market_stats.sort_values("mean_csat", ascending=False).reset_index(drop=True)

# ── Printed summary table ────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("PHASE 4 -- MARKET SUMMARY TABLE")
print("=" * 75)
header = (
    f"{'Market':<14} {'Count':>7}  {'Mean CSAT':>10}  "
    f"{'5* Rate':>8}  {'Below-4*':>9}  {'Misdelivery':>12}"
)
print(header)
print("-" * 75)
for _, row in market_stats.iterrows():
    print(
        f"{row['market']:<14} {int(row['count']):>7,}  {row['mean_csat']:>10.4f}  "
        f"{row['five_star_pct']:>7.2f}%  {row['below4_pct']:>8.2f}%  "
        f"{row['misdelivery_rate']:>10.3f}%"
    )
print("=" * 75)

# ── Colour gradient helper ───────────────────────────────────────────────────
# Interpolates from RED (worst = rank 0) to GREEN (best = top rank).
# Used so the bar chart colour encodes performance rank at a glance.

def rank_color(rank, total, best_color=GREEN, worst_color=RED):
    """Return a hex colour interpolated between worst_color and best_color."""
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    t   = rank / max(total - 1, 1)   # 0 = worst, 1 = best
    rgb_w = hex_to_rgb(worst_color)
    rgb_b = hex_to_rgb(best_color)
    blended = tuple(rgb_w[i] + t * (rgb_b[i] - rgb_w[i]) for i in range(3))
    return rgb_to_hex(blended)

# Assign colours: market_stats is sorted best->worst so index 0 = best.
n_markets = len(market_stats)
# Reverse rank so index 0 (best mean) gets rank = n-1 (GREEN end)
bar_colors_market = [
    rank_color(n_markets - 1 - i, n_markets)
    for i in range(n_markets)
]

# =============================================================================
# CHART 3 -- 03_csat_by_market.png
# Horizontal bar chart sorted best-at-top with GREEN->RED gradient and a
# vertical dashed line at the 4.9 target.
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))

y_pos   = range(n_markets)
markets = market_stats["market"].tolist()
means   = market_stats["mean_csat"].tolist()

bars = ax.barh(
    y_pos,
    means,
    color=bar_colors_market,
    edgecolor="#0c0f14",
    linewidth=0.6,
    height=0.55,
    zorder=3,
)

# Vertical dashed target line
ax.axvline(
    TARGET_CSAT,
    color=TEXT_MUTED,
    linewidth=1.3,
    linestyle="--",
    alpha=0.7,
    zorder=2,
)
ax.text(
    TARGET_CSAT + 0.002,
    n_markets - 0.55,
    f"Target {TARGET_CSAT}",
    color=TEXT_MUTED,
    fontsize=10,
    va="top",
)

# Annotate each bar with exact mean value
x_min = min(means) - 0.05
for bar, mean in zip(bars, means):
    x_right = bar.get_width()
    ax.text(
        x_right + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{mean:.4f}",
        va="center", ha="left",
        fontsize=11, color="#e8eaed", fontweight="bold",
    )

ax.set_yticks(list(y_pos))
ax.set_yticklabels(markets, fontsize=12)
ax.set_xlabel("Mean CSAT Score", fontsize=13, labelpad=8)
ax.set_title("Mean CSAT by Market  (best at top)", fontsize=15, fontweight="bold", pad=14)

# Tight x range so differences are visible
x_lo = max(0, min(means) - 0.12)
x_hi = max(means) + 0.10
ax.set_xlim(x_lo, x_hi)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.45, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "03_csat_by_market")
plt.close(fig)

# =============================================================================
# CHART 4 -- 04_market_multi_metric.png
# Three side-by-side subplots: Mean CSAT | 5-star Rate | Below-4-star Rate.
# Each subplot is a horizontal bar chart using the same colour gradient so the
# reader can spot whether a market's rank is consistent across all three views.
# =============================================================================

metrics = [
    ("mean_csat",    "Mean CSAT",      "#e8eaed", None),
    ("five_star_pct","5-Star Rate (%)", GREEN,     None),
    ("below4_pct",  "Below-4 Rate (%)", RED,       None),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig.suptitle(
    "Market Performance -- Three-Metric View",
    fontsize=15, fontweight="bold", y=1.02,
)

for ax, (col, label, bar_color, _) in zip(axes, metrics):
    vals    = market_stats[col].tolist()
    # For below4 the worst market is the one with the highest rate, so we
    # flip the gradient direction.
    if col == "below4_pct":
        colors = list(reversed(bar_colors_market))
    else:
        colors = bar_colors_market

    bars = ax.barh(
        list(y_pos),
        vals,
        color=colors,
        edgecolor="#0c0f14",
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )

    # Target line only on mean CSAT panel
    if col == "mean_csat":
        ax.axvline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2,
                   linestyle="--", alpha=0.65, zorder=2)

    # Value annotations
    fmt = ".4f" if col == "mean_csat" else ".1f"
    suffix = "" if col == "mean_csat" else "%"
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + (max(vals) - min(vals)) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:{fmt}}{suffix}",
            va="center", ha="left",
            fontsize=9, color="#e8eaed",
        )

    ax.set_xlabel(label, fontsize=11, labelpad=6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(markets, fontsize=11)
    ax.set_xlim(
        max(0, min(vals) - (max(vals) - min(vals)) * 0.15),
        max(vals) + (max(vals) - min(vals)) * 0.22,
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "04_market_multi_metric")
plt.close(fig)

# =============================================================================
# CHART 5 -- 05_misdelivery_by_market.png
# Bar chart of misdelivery rate (%) sorted descending. Each bar is annotated
# with "misdelivered / total" so the absolute count is immediately visible.
# =============================================================================

mis_sorted = market_stats.sort_values("misdelivery_rate", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 6))

mis_colors = [ACCENT if i == 0 else BLUE for i in range(len(mis_sorted))]

bars = ax.bar(
    mis_sorted["market"],
    mis_sorted["misdelivery_rate"],
    color=mis_colors,
    edgecolor="#0c0f14",
    linewidth=0.7,
    width=0.55,
    zorder=3,
)

# Annotate: rate on top, "misdelivered / total" just below
for bar, (_, row) in zip(bars, mis_sorted.iterrows()):
    y = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y + 0.003,
        f"{row['misdelivery_rate']:.3f}%",
        ha="center", va="bottom",
        fontsize=11, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y / 2,
        f"{int(row['misdelivered'])}/{int(row['total']):,}",
        ha="center", va="center",
        fontsize=9, color="#0c0f14", fontweight="bold",
    )

ax.set_xlabel("Market", fontsize=13, labelpad=8)
ax.set_ylabel("Misdelivery Rate (%)", fontsize=13, labelpad=8)
ax.set_title("Misdelivery Rate by Market", fontsize=15, fontweight="bold", pad=14)
ax.set_ylim(0, mis_sorted["misdelivery_rate"].max() * 1.35)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.45, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "05_misdelivery_by_market")
plt.close(fig)


# =============================================================================
# PHASE 5 -- TEMPORAL TRENDS
# Four charts: weekly aggregate, weekly per-market, day-of-week, and hour-of-
# day. The hour chart is the headline finding — it shows the late-night
# delivery cliff where CSAT crashes after midnight.
# =============================================================================

# ── Weekly aggregation ───────────────────────────────────────────────────────
weekly = (
    valid.groupby("week_number")["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reset_index()
    .sort_values("week_number")
)

# ── Day-of-week aggregation (ordered Mon→Sun) ────────────────────────────────
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
dow = (
    valid.groupby("day_of_week")["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reindex(DOW_ORDER)
    .reset_index()
)

# ── Hour-of-day aggregation ──────────────────────────────────────────────────
hourly = (
    valid.groupby("hour_of_day")["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reset_index()
    .sort_values("hour_of_day")
)

# ── Market-weekly aggregation ────────────────────────────────────────────────
mkt_weekly = (
    valid.groupby(["week_number", "market"])["csat_numeric"]
    .mean()
    .reset_index()
    .rename(columns={"csat_numeric": "mean_csat"})
)

# ── Weekly trend table ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 -- WEEKLY CSAT TREND")
print("=" * 55)
print(f"  {'Week':>6}  {'Deliveries':>12}  {'Mean CSAT':>10}")
print("-" * 55)
for _, row in weekly.iterrows():
    flag = " <-- trend start" if row["week_number"] == weekly["week_number"].min() else ""
    print(f"  {int(row['week_number']):>6}  {int(row['count']):>12,}  {row['mean_csat']:>10.4f}{flag}")
print("=" * 55)

# ── Critical time-of-day finding ────────────────────────────────────────────
night_hours  = hourly[hourly["hour_of_day"].isin([2, 4])]
day_hours    = hourly[hourly["hour_of_day"].between(9, 18)]
night_mean   = night_hours["mean_csat"].mean() if len(night_hours) else float("nan")
day_mean     = day_hours["mean_csat"].mean()   if len(day_hours)   else float("nan")
worst_hour_row = hourly.loc[hourly["mean_csat"].idxmin()]

print("\n" + "=" * 55)
print("PHASE 5 -- CRITICAL TIME-OF-DAY FINDING")
print("=" * 55)
print(f"  Daytime  CSAT (09-18h) avg : {day_mean:.4f}")
print(f"  Midnight CSAT (02h, 04h)   : {night_mean:.4f}")
print(f"  Worst single hour          : {int(worst_hour_row['hour_of_day']):02d}:00  "
      f"-> CSAT {worst_hour_row['mean_csat']:.4f}  "
      f"(n={int(worst_hour_row['count']):,})")
print(f"  CSAT cliff drop            : {day_mean - night_mean:.4f} points")
print("=" * 55)

# ── Market colours (consistent across all remaining charts) ─────────────────
MARKET_COLORS = {
    "Dallas":       "#34d399",   # GREEN
    "Denver":       "#60a5fa",   # BLUE
    "Chicago":      "#fbbf24",   # YELLOW
    "Baltimore":    "#f87171",   # RED
    "Philadelphia": "#a78bfa",   # PURPLE
}

# =============================================================================
# CHART 6 -- 06_csat_by_week.png
# Dual-axis: bars for delivery volume (right axis, muted) and a line for
# mean CSAT (left axis, ACCENT). A dashed target line at 4.9 sits on the
# left axis. Weeks 11-15 show the upward trend that the annotation calls out.
# =============================================================================

fig, ax1 = plt.subplots(figsize=(13, 7))
ax2 = ax1.twinx()

weeks  = weekly["week_number"].tolist()
counts = weekly["count"].tolist()
means  = weekly["mean_csat"].tolist()

# Volume bars on right axis (subtle — they contextualise the line)
ax2.bar(
    weeks, counts,
    color=BLUE, alpha=0.18, width=0.7, zorder=1, label="Deliveries"
)
ax2.set_ylabel("Delivery Volume", fontsize=12, color=TEXT_MUTED, labelpad=8)
ax2.tick_params(axis="y", colors=TEXT_MUTED)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(0, max(counts) * 2.2)   # push bars to lower half so line dominates

# Mean CSAT line on left axis
ax1.plot(
    weeks, means,
    color=ACCENT, linewidth=2.4, marker="o", markersize=6,
    markerfacecolor=ACCENT, markeredgecolor="#0c0f14", markeredgewidth=0.8,
    zorder=4, label="Mean CSAT",
)

# 4.9 target dashed line
ax1.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
            alpha=0.7, zorder=2)
ax1.text(
    weeks[-1] + 0.1, TARGET_CSAT + 0.003,
    f"Target {TARGET_CSAT}",
    color=TEXT_MUTED, fontsize=10, va="bottom", ha="right",
)

# Upward trend annotation spanning min to max week within 11-15
trend_weeks = [w for w in weeks if 11 <= w <= 15]
if len(trend_weeks) >= 2:
    tw_start = trend_weeks[0]
    tw_end   = trend_weeks[-1]
    m_start  = weekly.loc[weekly["week_number"] == tw_start, "mean_csat"].values[0]
    m_end    = weekly.loc[weekly["week_number"] == tw_end,   "mean_csat"].values[0]
    ax1.annotate(
        "",
        xy=(tw_end, m_end), xytext=(tw_start, m_start),
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
        zorder=5,
    )
    ax1.text(
        (tw_start + tw_end) / 2,
        max(m_start, m_end) + 0.008,
        f"Wk {tw_start}-{tw_end} uptrend",
        color=GREEN, fontsize=9, ha="center",
    )

# Value labels above each CSAT marker
for w, m in zip(weeks, means):
    ax1.text(w, m + 0.006, f"{m:.3f}", ha="center", va="bottom",
             fontsize=8, color="#e8eaed")

ax1.set_xlabel("ISO Week Number", fontsize=13, labelpad=8)
ax1.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax1.set_title("Weekly CSAT Trend vs Delivery Volume", fontsize=15,
              fontweight="bold", pad=14)
ax1.set_xticks(weeks)
ax1.set_xticklabels([f"Wk {w}" for w in weeks], fontsize=10, rotation=30, ha="right")
y_lo = min(means) - 0.04
y_hi = max(means) + 0.05
ax1.set_ylim(y_lo, y_hi)
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
           fontsize=10, framealpha=0.25)

plt.tight_layout()
save_fig(fig, "06_csat_by_week")
plt.close(fig)

# =============================================================================
# CHART 7 -- 07_csat_market_weekly.png
# One line per market, coloured by MARKET_COLORS. Weeks on x-axis. Legend
# placed below the chart so it doesn't obscure the lines.
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 7))

for market, grp in mkt_weekly.groupby("market"):
    grp = grp.sort_values("week_number")
    color = MARKET_COLORS.get(market, "#e8eaed")
    ax.plot(
        grp["week_number"], grp["mean_csat"],
        color=color, linewidth=2.2,
        marker="o", markersize=5.5,
        markerfacecolor=color, markeredgecolor="#0c0f14", markeredgewidth=0.7,
        label=market, zorder=3,
    )

ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
           alpha=0.65, zorder=2)
ax.text(
    mkt_weekly["week_number"].max() + 0.1,
    TARGET_CSAT + 0.003,
    f"Target {TARGET_CSAT}",
    color=TEXT_MUTED, fontsize=10, va="bottom", ha="right",
)

all_weeks = sorted(mkt_weekly["week_number"].unique())
ax.set_xticks(all_weeks)
ax.set_xticklabels([f"Wk {w}" for w in all_weeks], fontsize=10,
                   rotation=30, ha="right")
ax.set_xlabel("ISO Week Number", fontsize=13, labelpad=8)
ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title("Weekly CSAT by Market", fontsize=15, fontweight="bold", pad=14)

y_vals = mkt_weekly["mean_csat"]
ax.set_ylim(y_vals.min() - 0.06, y_vals.max() + 0.06)
ax.grid(linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.18),
    ncol=len(MARKET_COLORS), fontsize=11, framealpha=0.2,
)

plt.tight_layout()
save_fig(fig, "07_csat_market_weekly")
plt.close(fig)

# =============================================================================
# CHART 8 -- 08_csat_by_dow.png
# Monday-to-Sunday bar chart. Thursday bar = GREEN (best), Saturday = YELLOW
# (worst). All other days use BLUE. Value annotated above each bar.
# =============================================================================

best_dow  = dow.loc[dow["mean_csat"].idxmax(), "day_of_week"]
worst_dow = dow.loc[dow["mean_csat"].idxmin(), "day_of_week"]

dow_colors = []
for day in dow["day_of_week"]:
    if day == best_dow:
        dow_colors.append(GREEN)
    elif day == worst_dow:
        dow_colors.append(YELLOW)
    else:
        dow_colors.append(BLUE)

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    dow["day_of_week"], dow["mean_csat"],
    color=dow_colors, edgecolor="#0c0f14", linewidth=0.7,
    width=0.6, zorder=3,
)

for bar, (_, row) in zip(bars, dow.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{row['mean_csat']:.4f}",
        ha="center", va="bottom", fontsize=10, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() / 2,
        f"n={int(row['count']):,}",
        ha="center", va="center", fontsize=8, color="#0c0f14",
    )

# Legend patches
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=GREEN,  label=f"Best: {best_dow}"),
    Patch(facecolor=YELLOW, label=f"Worst: {worst_dow}"),
    Patch(facecolor=BLUE,   label="Other days"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=10, framealpha=0.25)

ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.1, linestyle="--",
           alpha=0.65, zorder=2)
ax.set_xlabel("Day of Week", fontsize=13, labelpad=8)
ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(f"Mean CSAT by Day of Week  (best: {best_dow}, worst: {worst_dow})",
             fontsize=14, fontweight="bold", pad=14)
y_lo = dow["mean_csat"].min() - 0.05
y_hi = dow["mean_csat"].max() + 0.04
ax.set_ylim(y_lo, y_hi)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "08_csat_by_dow")
plt.close(fig)

# =============================================================================
# CHART 9 -- 09_csat_by_hour.png  *** HEADLINE CHART ***
# Bars coloured by CSAT threshold tier. Right-axis line shows delivery volume
# per hour. Explicit drop annotations at the two worst late-night hours.
# =============================================================================

def hour_bar_color(mean):
    """Threshold-based colour for a single hour bar."""
    if mean >= 4.8:
        return GREEN
    elif mean >= 4.5:
        return YELLOW
    elif mean >= 4.0:
        return "#fb923c"   # orange
    else:
        return RED

fig, ax1 = plt.subplots(figsize=(15, 7))
ax2 = ax1.twinx()

hours  = hourly["hour_of_day"].tolist()
h_mean = hourly["mean_csat"].tolist()
h_cnt  = hourly["count"].tolist()

bar_colors_h = [hour_bar_color(m) for m in h_mean]

# Volume line on right axis
ax2.plot(
    hours, h_cnt,
    color=TEXT_MUTED, linewidth=1.4, linestyle="-",
    marker=".", markersize=5, alpha=0.6, zorder=2, label="Volume",
)
ax2.fill_between(hours, h_cnt, alpha=0.07, color=TEXT_MUTED, zorder=1)
ax2.set_ylabel("Delivery Volume", fontsize=12, color=TEXT_MUTED, labelpad=8)
ax2.tick_params(axis="y", colors=TEXT_MUTED)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(0, max(h_cnt) * 2.8)   # push to bottom half so bars dominate

# CSAT bars on left axis
bars = ax1.bar(
    hours, h_mean,
    color=bar_colors_h, edgecolor="#0c0f14", linewidth=0.5,
    width=0.7, zorder=3,
)

# 4.9 target line
ax1.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
            alpha=0.65, zorder=2)
ax1.text(
    hours[-1] + 0.3, TARGET_CSAT + 0.005,
    f"Target {TARGET_CSAT}", color=TEXT_MUTED, fontsize=9, va="bottom",
)

# Value labels above each bar
for h, m in zip(hours, h_mean):
    ax1.text(h, m + 0.008, f"{m:.2f}", ha="center", va="bottom",
             fontsize=7.5, color="#e8eaed")

# ── Annotation at extreme late-night drop points ─────────────────────────────
drop_hours = [2, 4]
for dh in drop_hours:
    row = hourly[hourly["hour_of_day"] == dh]
    if row.empty:
        continue
    dh_mean = row["mean_csat"].values[0]
    dh_cnt  = row["count"].values[0]
    ax1.annotate(
        f"{dh:02d}:00\nCSAT {dh_mean:.2f}\n(n={int(dh_cnt):,})",
        xy=(dh, dh_mean),
        xytext=(dh + (2 if dh < 12 else -2), dh_mean - 0.18),
        fontsize=8.5, color=RED, fontweight="bold",
        ha="left" if dh < 12 else "right",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.3),
        zorder=6,
    )

ax1.set_xlabel("Hour of Day (UTC from updated_at)", fontsize=13, labelpad=8)
ax1.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax1.set_title(
    "Mean CSAT by Hour of Day  -- Late-Night Delivery Cliff",
    fontsize=15, fontweight="bold", pad=14,
)
ax1.set_xticks(hours)
ax1.set_xticklabels([f"{h:02d}h" for h in hours], fontsize=9, rotation=45, ha="right")
y_lo = max(0, min(h_mean) - 0.25)
y_hi = max(h_mean) + 0.12
ax1.set_ylim(y_lo, y_hi)
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# Colour-tier legend
from matplotlib.patches import Patch as _Patch
tier_legend = [
    _Patch(facecolor=GREEN,    label=">=4.8  Healthy"),
    _Patch(facecolor=YELLOW,   label=">=4.5  Caution"),
    _Patch(facecolor="#fb923c",label=">=4.0  Warning"),
    _Patch(facecolor=RED,      label="< 4.0  Critical"),
]
ax1.legend(handles=tier_legend, loc="lower left", fontsize=9, framealpha=0.25)

lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, loc="upper right", fontsize=9, framealpha=0.25)

plt.tight_layout()
save_fig(fig, "09_csat_by_hour")
plt.close(fig)


# =============================================================================
# PHASE 6 -- DRIVER ANALYSIS
# Per-driver aggregation with a minimum-volume filter so one-delivery flukes
# don't distort the distribution. Charts expose the tail of low performers
# and the classic variance-shrinkage pattern as volume grows.
# =============================================================================

MIN_DELIVERIES = 10   # qualification threshold

# ── Per-driver aggregation (all valid-CSAT rows) ─────────────────────────────
driver_raw = (
    valid.groupby("driver_id")
    .agg(
        deliveries  =("csat_numeric", "count"),
        mean_csat   =("csat_numeric", "mean"),
        five_star_n =("csat_numeric", lambda s: (s == 5).sum()),
        below4_n    =("csat_numeric", lambda s: (s < 4).sum()),
        # Majority market for this driver (mode)
        market      =("market", lambda s: s.mode().iloc[0]),
    )
    .reset_index()
)

driver_raw["five_star_pct"] = driver_raw["five_star_n"] / driver_raw["deliveries"] * 100
driver_raw["below4_pct"]   = driver_raw["below4_n"]    / driver_raw["deliveries"] * 100

# Qualified subset
qualified = driver_raw[driver_raw["deliveries"] >= MIN_DELIVERIES].copy()
qualified = qualified.sort_values("mean_csat", ascending=True).reset_index(drop=True)

total_drivers    = driver_raw["driver_id"].nunique()
qual_count       = len(qualified)
below45_count    = (qualified["mean_csat"] < 4.5).sum()
below40_count    = (qualified["mean_csat"] < 4.0).sum()

# ── Printed summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 6 -- DRIVER ANALYSIS SUMMARY")
print("=" * 65)
print(f"  Total unique drivers         : {total_drivers:,}")
print(f"  Qualified (>={MIN_DELIVERIES} deliveries)  : {qual_count:,}")
print(f"  Qualified below 4.5 CSAT     : {below45_count:,}  "
      f"({below45_count/qual_count*100:.1f}% of qualified)")
print(f"  Qualified below 4.0 CSAT     : {below40_count:,}  "
      f"({below40_count/qual_count*100:.1f}% of qualified)")

print(f"\n  {'Rank':<5} {'Driver ID':<24} {'Market':<14} "
      f"{'Deliveries':>11} {'Mean CSAT':>10} {'Below-4':>8}")
print("  " + "-" * 75)
bottom10 = qualified.head(10)
for rank, (_, row) in enumerate(bottom10.iterrows(), 1):
    print(
        f"  {rank:<5} {str(row['driver_id']):<24} {row['market']:<14} "
        f"{int(row['deliveries']):>11,} {row['mean_csat']:>10.4f} "
        f"{row['below4_pct']:>7.1f}%"
    )
print("=" * 65)

# =============================================================================
# CHART 10 -- 10_driver_csat_histogram.png
# Distribution of per-driver mean CSAT for qualified drivers. Red shading left
# of 4.5 flags the problem tail. Vertical lines for overall mean and target.
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

q_means = qualified["mean_csat"].values

# Full histogram in BLUE
n, bins, patches = ax.hist(
    q_means, bins=40,
    color=BLUE, edgecolor="#0c0f14", linewidth=0.4,
    alpha=0.85, zorder=3,
)

# Re-colour bars below 4.5 threshold in RED
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge < 4.5:
        patch.set_facecolor(RED)
        patch.set_alpha(0.75)

# Shaded region label
ax.axvspan(
    q_means.min() - 0.05, 4.5,
    color=RED, alpha=0.08, zorder=1, label="Below 4.5 (problematic)",
)

# Overall mean line
overall_driver_mean = qualified["mean_csat"].mean()
ax.axvline(overall_driver_mean, color=ACCENT, linewidth=1.8, linestyle="-",
           zorder=4, label=f"Fleet mean  {overall_driver_mean:.4f}")
ax.text(overall_driver_mean + 0.005, ax.get_ylim()[1] * 0.92,
        f"Fleet\nmean\n{overall_driver_mean:.3f}",
        color=ACCENT, fontsize=9, va="top")

# 4.9 target line
ax.axvline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.4, linestyle="--",
           zorder=4, label=f"Target {TARGET_CSAT}")
ax.text(TARGET_CSAT + 0.005, ax.get_ylim()[1] * 0.75,
        f"Target\n{TARGET_CSAT}",
        color=TEXT_MUTED, fontsize=9, va="top")

# 4.5 boundary tick
ax.axvline(4.5, color=RED, linewidth=1.0, linestyle=":", alpha=0.6, zorder=3)

ax.set_xlabel("Driver Mean CSAT", fontsize=13, labelpad=8)
ax.set_ylabel("Number of Drivers", fontsize=13, labelpad=8)
ax.set_title(
    f"Driver CSAT Distribution  (qualified: n>={MIN_DELIVERIES}, "
    f"n={qual_count:,} drivers)",
    fontsize=14, fontweight="bold", pad=14,
)
ax.legend(fontsize=10, framealpha=0.25, loc="upper left")
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "10_driver_csat_histogram")
plt.close(fig)

# =============================================================================
# CHART 11 -- 11_bottom_drivers.png
# Horizontal bar chart for the 15 lowest-mean qualified drivers. Bars are
# coloured by their home market so regional patterns are visible at a glance.
# Delivery count is annotated inside/beside each bar.
# =============================================================================

bottom15 = qualified.head(15).copy()
# Re-sort so worst is at bottom of horizontal chart (ascending = top is best)
bottom15 = bottom15.sort_values("mean_csat", ascending=True).reset_index(drop=True)

b15_colors = [MARKET_COLORS.get(m, "#e8eaed") for m in bottom15["market"]]

fig, ax = plt.subplots(figsize=(12, 7))

y_pos15 = range(len(bottom15))
bars = ax.barh(
    list(y_pos15),
    bottom15["mean_csat"],
    color=b15_colors,
    edgecolor="#0c0f14",
    linewidth=0.5,
    height=0.65,
    zorder=3,
)

# 4.9 and 4.5 reference lines
ax.axvline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
           alpha=0.65, zorder=2, label=f"Target {TARGET_CSAT}")
ax.axvline(4.5, color=RED, linewidth=1.0, linestyle=":", alpha=0.55,
           zorder=2, label="4.5 threshold")

# Annotate: mean CSAT at bar end, delivery count inside bar
for bar, (_, row) in zip(bars, bottom15.iterrows()):
    w = bar.get_width()
    ax.text(
        w + 0.005, bar.get_y() + bar.get_height() / 2,
        f"{w:.4f}",
        va="center", ha="left", fontsize=9, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        max(w * 0.5, ax.get_xlim()[0] + 0.02),
        bar.get_y() + bar.get_height() / 2,
        f"n={int(row['deliveries'])}",
        va="center", ha="center", fontsize=8, color="#0c0f14", fontweight="bold",
    )

# Y-tick labels: truncated driver ID
driver_labels = [str(d)[:16] for d in bottom15["driver_id"]]
ax.set_yticks(list(y_pos15))
ax.set_yticklabels(driver_labels, fontsize=9)

# Market colour legend
from matplotlib.patches import Patch as _P2
mkt_legend = [
    _P2(facecolor=MARKET_COLORS.get(m, "#e8eaed"), label=m)
    for m in sorted(bottom15["market"].unique())
]
ax.legend(
    handles=mkt_legend,
    loc="lower right", fontsize=9, framealpha=0.25, title="Market",
)

x_lo = max(0, bottom15["mean_csat"].min() - 0.2)
x_hi = TARGET_CSAT + 0.08
ax.set_xlim(x_lo, x_hi)
ax.set_xlabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(
    f"Bottom 15 Qualified Drivers by Mean CSAT  (>={MIN_DELIVERIES} deliveries)",
    fontsize=14, fontweight="bold", pad=14,
)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "11_bottom_drivers")
plt.close(fig)

# =============================================================================
# CHART 12 -- 12_driver_scatter.png
# Scatter plot: x=delivery count, y=mean CSAT, one point per qualified driver,
# coloured by market. The funnel shape — wide variance at low volume, tight
# at high volume — is the key pattern to show.
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

for market, grp in qualified.groupby("market"):
    color = MARKET_COLORS.get(market, "#e8eaed")
    ax.scatter(
        grp["deliveries"], grp["mean_csat"],
        color=color, alpha=0.5, s=40,
        edgecolors="#0c0f14", linewidths=0.3,
        label=market, zorder=3,
    )

# Reference lines
ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.2, linestyle="--",
           alpha=0.65, zorder=2, label=f"Target {TARGET_CSAT}")
ax.axhline(4.5, color=RED, linewidth=1.0, linestyle=":", alpha=0.55,
           zorder=2, label="4.5 threshold")
ax.axhline(overall_driver_mean, color=ACCENT, linewidth=1.2, linestyle="-",
           alpha=0.6, zorder=2, label=f"Fleet mean {overall_driver_mean:.3f}")

# Shade the low-volume high-variance region
ax.axvspan(
    MIN_DELIVERIES, 30,
    color=YELLOW, alpha=0.05, zorder=1, label="High-variance zone (<30 del.)",
)

ax.set_xlabel("Delivery Count (qualified drivers)", fontsize=13, labelpad=8)
ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(
    "Driver CSAT vs Delivery Volume  -- Variance Shrinks with Volume",
    fontsize=14, fontweight="bold", pad=14,
)
ax.legend(
    loc="lower right", fontsize=9, framealpha=0.25, ncol=2,
)
ax.grid(linestyle=":", linewidth=0.5, alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

y_lo = max(0, qualified["mean_csat"].min() - 0.2)
ax.set_ylim(y_lo, qualified["mean_csat"].max() + 0.1)

plt.tight_layout()
save_fig(fig, "12_driver_scatter")
plt.close(fig)


# =============================================================================
# PHASE 7 -- OPERATIONS
# Four charts linking operational variables (duration, instructions presence,
# delivery outcome, route size) to CSAT. Each answers a "what can ops change?"
# question rather than just describing the distribution.
# =============================================================================

# ── Route-size feature ────────────────────────────────────────────────────────
# Count packages per route, then join back onto the valid-CSAT rows so every
# package row knows how large its route was on that day.
route_size = (
    df.groupby("route_id")["package_id"]
    .count()
    .rename("route_pkg_count")
    .reset_index()
)
valid_ops = valid.merge(route_size, on="route_id", how="left")

BUCKET_BINS   = [0, 3, 6, 9, 12, 999]
BUCKET_LABELS = ["1-3", "4-6", "7-9", "10-12", "13+"]
valid_ops["route_bucket"] = pd.cut(
    valid_ops["route_pkg_count"],
    bins=BUCKET_BINS,
    labels=BUCKET_LABELS,
)

# ── 1. Duration vs CSAT aggregation ──────────────────────────────────────────
# delivery_duration_hrs was already computed and winsorised to 0-24 in Phase 2.
dur_by_csat = (
    valid_ops[valid_ops["delivery_duration_hrs"].notna()]
    .groupby("csat_numeric")["delivery_duration_hrs"]
    .agg(mean_dur="mean", count="count")
    .reset_index()
    .sort_values("csat_numeric")
)

# ── 2. Instructions impact aggregation ───────────────────────────────────────
valid_ops["has_instructions"] = valid_ops["instructions"].notna()
instr_agg = (
    valid_ops.groupby("has_instructions")["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reset_index()
)
instr_agg["label"] = instr_agg["has_instructions"].map(
    {True: "With Instructions", False: "Without Instructions"}
)
# Ensure consistent order: Without first, With second
instr_agg = instr_agg.sort_values("has_instructions").reset_index(drop=True)

csat_without = instr_agg.loc[instr_agg["has_instructions"] == False, "mean_csat"].values[0]
csat_with    = instr_agg.loc[instr_agg["has_instructions"] == True,  "mean_csat"].values[0]
instr_gap    = csat_with - csat_without

# ── 3. Misdelivery impact aggregation ────────────────────────────────────────
misdelivery_csat = (
    valid_ops.groupby("last_event")["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reset_index()
)

# ── 4. Route-size vs CSAT aggregation ────────────────────────────────────────
route_csat = (
    valid_ops.groupby("route_bucket", observed=True)["csat_numeric"]
    .agg(mean_csat="mean", count="count")
    .reset_index()
)

# ── Printed insights ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 7 -- OPERATIONS INSIGHTS")
print("=" * 65)

print("\n  [Duration vs CSAT]")
for _, row in dur_by_csat.iterrows():
    print(f"    CSAT {int(row['csat_numeric'])}*  mean duration: "
          f"{row['mean_dur']:.2f} hrs  (n={int(row['count']):,})")
dur_1 = dur_by_csat.loc[dur_by_csat["csat_numeric"] == 1, "mean_dur"]
dur_5 = dur_by_csat.loc[dur_by_csat["csat_numeric"] == 5, "mean_dur"]
if len(dur_1) and len(dur_5):
    print(f"    -> 1-star routes run {dur_1.values[0] - dur_5.values[0]:.2f} hrs "
          f"longer on average than 5-star routes.")

print("\n  [Instructions Impact]")
for _, row in instr_agg.iterrows():
    print(f"    {row['label']:<22}: CSAT {row['mean_csat']:.4f}  (n={int(row['count']):,})")
print(f"    -> Gap: {instr_gap:+.4f}  "
      f"({'with' if instr_gap > 0 else 'without'} instructions scores higher)")

print("\n  [Misdelivery Impact]")
for _, row in misdelivery_csat.iterrows():
    print(f"    {row['last_event']:<14}: CSAT {row['mean_csat']:.4f}  "
          f"(n={int(row['count']):,})")
md_del = misdelivery_csat.loc[misdelivery_csat["last_event"] == "delivered",    "mean_csat"].values
md_mis = misdelivery_csat.loc[misdelivery_csat["last_event"] == "misdelivered", "mean_csat"].values
if len(md_del) and len(md_mis):
    print(f"    -> Misdelivery destroys {md_del[0] - md_mis[0]:.2f} CSAT points.")

print("\n  [Route Size vs CSAT]")
for _, row in route_csat.iterrows():
    print(f"    {str(row['route_bucket']):<8} pkgs: CSAT {row['mean_csat']:.4f}  "
          f"(n={int(row['count']):,})")

print("=" * 65)

# =============================================================================
# CHART 13 -- 13_duration_vs_csat.png
# Bar chart: mean delivery duration by CSAT rating 1-5. CSAT mapped to its
# star-tier colour. Longer routes correlating with lower satisfaction is the
# key narrative.
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

dur_colors = [BAR_COLORS[int(r)] for r in dur_by_csat["csat_numeric"]]

bars = ax.bar(
    dur_by_csat["csat_numeric"].astype(int),
    dur_by_csat["mean_dur"],
    color=dur_colors, edgecolor="#0c0f14", linewidth=0.7,
    width=0.6, zorder=3,
)

for bar, (_, row) in zip(bars, dur_by_csat.iterrows()):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.01,
        f"{h:.2f} hrs",
        ha="center", va="bottom", fontsize=10,
        color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2, h / 2,
        f"n={int(row['count']):,}",
        ha="center", va="center", fontsize=8, color="#0c0f14",
    )

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], fontsize=11)
ax.set_xlabel("CSAT Rating", fontsize=13, labelpad=8)
ax.set_ylabel("Mean Delivery Duration (hours)", fontsize=13, labelpad=8)
ax.set_title(
    "Mean Delivery Duration by CSAT Rating\n(longer routes correlate with lower satisfaction)",
    fontsize=14, fontweight="bold", pad=14,
)
ax.set_ylim(0, dur_by_csat["mean_dur"].max() * 1.22)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# Trend arrow from 5-star bar to 1-star bar
if len(dur_1) and len(dur_5):
    ax.annotate(
        "",
        xy=(1, dur_1.values[0] * 0.98), xytext=(5, dur_5.values[0] * 1.02),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, linestyle="dashed"),
        zorder=5,
    )
    mid_y = (dur_1.values[0] + dur_5.values[0]) / 2
    ax.text(3, mid_y + 0.05,
            f"+{dur_1.values[0] - dur_5.values[0]:.2f} hrs for 1-star",
            ha="center", fontsize=9, color=ACCENT)

plt.tight_layout()
save_fig(fig, "13_duration_vs_csat")
plt.close(fig)

# =============================================================================
# CHART 14 -- 14_instructions_impact.png
# Two side-by-side bars. The gap annotation is the headline number.
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 6))

bar_colors_instr = [BLUE, GREEN]
bars = ax.bar(
    instr_agg["label"], instr_agg["mean_csat"],
    color=bar_colors_instr, edgecolor="#0c0f14", linewidth=0.7,
    width=0.45, zorder=3,
)

for bar, (_, row) in zip(bars, instr_agg.iterrows()):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.002,
        f"{h:.4f}",
        ha="center", va="bottom", fontsize=12, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2, h * 0.5,
        f"n={int(row['count']):,}",
        ha="center", va="center", fontsize=9, color="#0c0f14",
    )

# Gap bracket annotation between the two bars
x0 = bars[0].get_x() + bars[0].get_width()
x1 = bars[1].get_x()
y_bracket = max(instr_agg["mean_csat"]) + 0.012
ax.annotate(
    "", xy=(x1, y_bracket), xytext=(x0, y_bracket),
    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6),
)
ax.text(
    (x0 + x1) / 2, y_bracket + 0.003,
    f"Gap: {instr_gap:+.4f}",
    ha="center", va="bottom", fontsize=11, color=ACCENT, fontweight="bold",
)

ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(
    "CSAT Impact of Delivery Instructions\n(packages with instructions score higher)",
    fontsize=14, fontweight="bold", pad=14,
)
y_lo = min(instr_agg["mean_csat"]) - 0.06
y_hi = max(instr_agg["mean_csat"]) + 0.06
ax.set_ylim(y_lo, y_hi)
ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.1, linestyle="--",
           alpha=0.6, zorder=2)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "14_instructions_impact")
plt.close(fig)

# =============================================================================
# CHART 15 -- 15_misdelivery_impact.png
# Two bars only — the dramatic CSAT collapse on misdelivery is self-evident.
# =============================================================================

# Sort so "delivered" is first
mis_plot = misdelivery_csat.sort_values("last_event", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))

bar_colors_mis = [GREEN if e == "delivered" else RED
                  for e in mis_plot["last_event"]]

bars = ax.bar(
    mis_plot["last_event"].str.capitalize(),
    mis_plot["mean_csat"],
    color=bar_colors_mis, edgecolor="#0c0f14", linewidth=0.7,
    width=0.45, zorder=3,
)

for bar, (_, row) in zip(bars, mis_plot.iterrows()):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.03,
        f"{h:.4f}",
        ha="center", va="bottom", fontsize=13, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2, h / 2,
        f"n={int(row['count']):,}",
        ha="center", va="center", fontsize=10, color="#0c0f14", fontweight="bold",
    )

# Annotate the drop
if len(md_del) and len(md_mis):
    drop = md_del[0] - md_mis[0]
    ax.annotate(
        f"-{drop:.2f} pts",
        xy=(1, md_mis[0] + 0.08),
        xytext=(0.5, (md_del[0] + md_mis[0]) / 2),
        fontsize=12, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.8),
        zorder=5,
    )

ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.1, linestyle="--",
           alpha=0.6, zorder=2)
ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(
    "CSAT: Delivered vs Misdelivered\n(misdelivery is the single largest CSAT driver)",
    fontsize=14, fontweight="bold", pad=14,
)
ax.set_ylim(0, md_del[0] + 0.5 if len(md_del) else 5.5)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "15_misdelivery_impact")
plt.close(fig)

# =============================================================================
# CHART 16 -- 16_route_size_vs_csat.png
# Bar chart with route-size buckets on x-axis. Colour encodes the same
# threshold tiers as the hour chart so the reader has a consistent visual
# language across the deck.
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))

rc_colors = [hour_bar_color(m) for m in route_csat["mean_csat"]]

bars = ax.bar(
    route_csat["route_bucket"].astype(str),
    route_csat["mean_csat"],
    color=rc_colors, edgecolor="#0c0f14", linewidth=0.7,
    width=0.55, zorder=3,
)

for bar, (_, row) in zip(bars, route_csat.iterrows()):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.002,
        f"{h:.4f}",
        ha="center", va="bottom", fontsize=10, color="#e8eaed", fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2, h / 2,
        f"n={int(row['count']):,}",
        ha="center", va="center", fontsize=8, color="#0c0f14",
    )

ax.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.1, linestyle="--",
           alpha=0.65, zorder=2)
ax.text(
    len(route_csat) - 0.48, TARGET_CSAT + 0.002,
    f"Target {TARGET_CSAT}", color=TEXT_MUTED, fontsize=9, va="bottom",
)

# Threshold-tier legend (reuse tier_legend from Phase 5)
from matplotlib.patches import Patch as _P3
rc_legend = [
    _P3(facecolor=GREEN,    label=">=4.8  Healthy"),
    _P3(facecolor=YELLOW,   label=">=4.5  Caution"),
    _P3(facecolor="#fb923c",label=">=4.0  Warning"),
    _P3(facecolor=RED,      label="< 4.0  Critical"),
]
ax.legend(handles=rc_legend, loc="lower left", fontsize=9, framealpha=0.25)

ax.set_xlabel("Route Size (packages per route)", fontsize=13, labelpad=8)
ax.set_ylabel("Mean CSAT", fontsize=13, labelpad=8)
ax.set_title(
    "Mean CSAT by Route Size\n(larger routes trend toward lower satisfaction)",
    fontsize=14, fontweight="bold", pad=14,
)
y_lo = route_csat["mean_csat"].min() - 0.05
y_hi = route_csat["mean_csat"].max() + 0.05
ax.set_ylim(y_lo, y_hi)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "16_route_size_vs_csat")
plt.close(fig)


# =============================================================================
# PHASE 8 -- CUSTOMER VOICE ANALYSIS
# Text mining on low-CSAT comments (rating <= 3 with non-null comment).
# Two charts: keyword frequency bar and theme donut.
# =============================================================================

# ── Low-CSAT comment corpus ───────────────────────────────────────────────────
low_csat = valid[
    (valid["csat_numeric"] <= 3) & (valid["csat_comment"].notna())
].copy()

print("\n" + "=" * 60)
print("PHASE 8 -- LOW-CSAT COMMENT CORPUS")
print("=" * 60)
print(f"  Low-CSAT rows (rating<=3 with comment): {len(low_csat):,}")
print(f"\n  15 sample comments:")
sample_comments = low_csat["csat_comment"].sample(
    min(15, len(low_csat)), random_state=7
).tolist()
for i, c in enumerate(sample_comments, 1):
    print(f"  {i:>2}. {str(c)[:110]}")

# ── Tokenisation & stopword removal ──────────────────────────────────────────
STOPWORDS = set([
    "the","a","an","is","was","it","to","and","of","in","for","my","i","we",
    "not","that","this","but","with","have","had","are","be","on","at","from",
    "or","so","if","just","been","has","very","can","do","did","no","me","our",
    "you","your","they","them","would","will","should","could","its","im",
    "dont","ive","doesnt","were","about","all","one","up","out","get","got",
    "also","than","when","more",
])

def tokenize(text):
    """Lowercase, split on non-alpha, drop stopwords and single chars."""
    tokens = re.split(r"[^a-zA-Z]+", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

all_tokens = []
for comment in low_csat["csat_comment"]:
    all_tokens.extend(tokenize(comment))

word_freq = pd.Series(all_tokens).value_counts()
top20 = word_freq.head(20)

# ── Theme classification ──────────────────────────────────────────────────────
# Rules applied in priority order; first match wins.
THEME_RULES = [
    ("Wrong Address",          r"wrong|address|house|misdeliver|not my|not mine"),
    ("Instructions Not Followed", r"door|instruction|direction|left|place|front|behind|leave"),
    ("Late Delivery",          r"late|time|early|hours|long|wait"),
    ("Driver Behavior",        r"rude|driver|behavior|attitude|shouted|expletive"),
    ("Package Condition",      r"cold|warm|damaged|leak|broken|condition"),
    ("Missing/Stolen",         r"stolen|missing|never|no delivery|not delivered"),
]

def classify_theme(comment):
    s = str(comment).lower()
    for theme, pattern in THEME_RULES:
        if re.search(pattern, s):
            return theme
    return "Other"

low_csat["theme"] = low_csat["csat_comment"].apply(classify_theme)
theme_counts = low_csat["theme"].value_counts()

print(f"\n  Theme distribution ({len(low_csat):,} comments):")
for theme, cnt in theme_counts.items():
    pct = cnt / len(low_csat) * 100
    print(f"    {theme:<30} {cnt:>5,}  ({pct:.1f}%)")
print("=" * 60)

# =============================================================================
# CHART 17 -- 17_low_csat_keywords.png
# Horizontal bar chart of top-20 word frequencies. Bars are coloured by a
# continuous RED gradient (darkest = most frequent) so the visual weight sits
# on the words that matter most.
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

words  = top20.index.tolist()
counts = top20.values.tolist()
n_bars = len(words)

# Gradient: most-frequent word at full RED opacity, fading to muted pink
bar_alphas  = [0.95 - 0.45 * i / max(n_bars - 1, 1) for i in range(n_bars)]
word_colors = [
    matplotlib.colors.to_hex(
        matplotlib.colors.to_rgba(RED, alpha=a)[:3]   # drop alpha — use RGB only
    )
    for a in bar_alphas
]
# Blend toward dark background instead of pure white fade
def blend_to_bg(hex_color, bg="#151920", t=0.0):
    """Interpolate hex_color toward bg by factor t (0=full color, 1=bg)."""
    def h2r(h):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    c  = h2r(hex_color)
    b  = h2r(bg)
    r  = [c[i] * (1 - t) + b[i] * t for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(int(r[0]*255), int(r[1]*255), int(r[2]*255))

word_colors = [blend_to_bg(RED, t=0.5 * i / max(n_bars - 1, 1)) for i in range(n_bars)]

# Reverse so highest bar is at top
words_rev  = list(reversed(words))
counts_rev = list(reversed(counts))
colors_rev = list(reversed(word_colors))

bars = ax.barh(
    words_rev, counts_rev,
    color=colors_rev, edgecolor="#0c0f14", linewidth=0.4,
    height=0.7, zorder=3,
)

for bar, cnt in zip(bars, counts_rev):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        str(cnt),
        va="center", ha="left", fontsize=9, color="#e8eaed",
    )

ax.set_xlabel("Word Frequency", fontsize=13, labelpad=8)
ax.set_title(
    f"Top 20 Keywords in Low-CSAT Comments (rating <= 3,  n={len(low_csat):,})",
    fontsize=14, fontweight="bold", pad=14,
)
ax.set_xlim(0, max(counts_rev) * 1.15)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save_fig(fig, "17_low_csat_keywords")
plt.close(fig)

# =============================================================================
# CHART 18 -- 18_comment_themes.png
# Donut chart. Each wedge is labelled with theme name + count + percentage.
# A custom colour palette distinguishes the themes; the hole text shows the
# total comment count so the absolute scale is always visible.
# =============================================================================

THEME_COLORS = {
    "Wrong Address":              RED,
    "Instructions Not Followed":  "#fb923c",   # orange
    "Late Delivery":              YELLOW,
    "Driver Behavior":            PURPLE,
    "Package Condition":          BLUE,
    "Missing/Stolen":             "#f43f5e",   # rose
    "Other":                      TEXT_MUTED,
}

theme_labels = theme_counts.index.tolist()
theme_vals   = theme_counts.values.tolist()
theme_colors_donut = [THEME_COLORS.get(t, TEXT_MUTED) for t in theme_labels]
theme_total  = sum(theme_vals)

fig, ax = plt.subplots(figsize=(11, 8))

wedges, texts, autotexts = ax.pie(
    theme_vals,
    labels=None,
    colors=theme_colors_donut,
    autopct=lambda p: f"{p:.1f}%",
    pctdistance=0.78,
    startangle=140,
    wedgeprops=dict(width=0.52, edgecolor="#0c0f14", linewidth=1.0),
)

for at in autotexts:
    at.set_fontsize(10)
    at.set_color("#e8eaed")
    at.set_fontweight("bold")

# Custom legend with count + pct
legend_labels = [
    f"{t}  ({c:,}  {c/theme_total*100:.1f}%)"
    for t, c in zip(theme_labels, theme_vals)
]
ax.legend(
    wedges, legend_labels,
    loc="center left", bbox_to_anchor=(0.88, 0.5),
    fontsize=10, framealpha=0.2,
    title="Theme", title_fontsize=10,
)

# Centre hole text
ax.text(
    0, 0,
    f"{theme_total:,}\ncomments",
    ha="center", va="center",
    fontsize=13, color="#e8eaed", fontweight="bold",
    linespacing=1.5,
)

ax.set_title(
    "Low-CSAT Comment Themes  (rating <= 3)",
    fontsize=15, fontweight="bold", pad=18,
)

plt.tight_layout()
save_fig(fig, "18_comment_themes")
plt.close(fig)


# =============================================================================
# PHASE 9 -- SUMMARY DASHBOARD
# Single 18x22 figure, 3x2 grid. Every subplot re-uses data already computed
# in earlier phases — no new aggregations needed.
# =============================================================================

# ── Pre-compute KPI scalars ───────────────────────────────────────────────────
kpi_mean_csat      = valid["csat_numeric"].mean()
kpi_5star_pct      = (valid["csat_numeric"] == 5).sum() / len(valid) * 100
kpi_below4_pct     = (valid["csat_numeric"] < 4).sum()  / len(valid) * 100
kpi_misdelivery    = (df["last_event"] == "misdelivered").sum() / len(df) * 100
kpi_total_drivers  = driver_raw["driver_id"].nunique()

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor("#0c0f14")

gs = fig.add_gridspec(
    3, 2,
    hspace=0.38, wspace=0.30,
    left=0.07, right=0.96,
    top=0.93,  bottom=0.04,
)

ax_kpi    = fig.add_subplot(gs[0, 0])   # top-left    : KPI text panel
ax_dist   = fig.add_subplot(gs[0, 1])   # top-right   : rating distribution
ax_market = fig.add_subplot(gs[1, 0])   # mid-left    : CSAT by market
ax_weekly = fig.add_subplot(gs[1, 1])   # mid-right   : weekly trend
ax_hour   = fig.add_subplot(gs[2, 0])   # bottom-left : hour-of-day cliff
ax_instr  = fig.add_subplot(gs[2, 1])   # bottom-right: instructions impact

DARK_AX   = "#151920"
EDGE_COL  = "#2a2f3a"

for ax in [ax_kpi, ax_dist, ax_market, ax_weekly, ax_hour, ax_instr]:
    ax.set_facecolor(DARK_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE_COL)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 1 — KPI CARD (text only)
# ─────────────────────────────────────────────────────────────────────────────
ax_kpi.set_xlim(0, 1)
ax_kpi.set_ylim(0, 1)
ax_kpi.axis("off")

# Card background rectangle
from matplotlib.patches import FancyBboxPatch
card = FancyBboxPatch(
    (0.04, 0.04), 0.92, 0.92,
    boxstyle="round,pad=0.02",
    linewidth=1.2, edgecolor=EDGE_COL,
    facecolor="#1a1f2e",
    transform=ax_kpi.transAxes, zorder=1,
    clip_on=False,
)
ax_kpi.add_patch(card)

ax_kpi.text(
    0.5, 0.88, "KEY PERFORMANCE INDICATORS",
    transform=ax_kpi.transAxes,
    ha="center", va="center",
    fontsize=10, color=TEXT_MUTED, fontweight="bold", zorder=2,
)

# Divider line under header
ax_kpi.axhline(
    0.81, xmin=0.06, xmax=0.94,
    color=EDGE_COL, linewidth=0.8, zorder=2,
)

kpi_rows = [
    # (label,         value_str,                          value_color)
    ("Overall CSAT",  f"{kpi_mean_csat:.3f}",             ACCENT),
    ("Target",        f"{TARGET_CSAT}",                   TEXT_MUTED),
    ("5-Star Rate",   f"{kpi_5star_pct:.1f}%",            GREEN),
    ("Below 4-Star",  f"{kpi_below4_pct:.1f}%",           RED),
    ("Misdelivery",   f"{kpi_misdelivery:.3f}%",          YELLOW),
    ("Total Drivers", f"{kpi_total_drivers:,}",           BLUE),
]

y_positions = [0.70, 0.59, 0.48, 0.37, 0.26, 0.15]

for (label, value, color), y in zip(kpi_rows, y_positions):
    ax_kpi.text(
        0.12, y, label,
        transform=ax_kpi.transAxes,
        ha="left", va="center",
        fontsize=11, color=TEXT_MUTED, zorder=2,
    )
    ax_kpi.text(
        0.88, y, value,
        transform=ax_kpi.transAxes,
        ha="right", va="center",
        fontsize=16, color=color, fontweight="bold", zorder=2,
    )
    # Subtle row separator
    if y != y_positions[-1]:
        ax_kpi.axhline(
            y - 0.055, xmin=0.06, xmax=0.94,
            color=EDGE_COL, linewidth=0.4, alpha=0.6, zorder=2,
        )

ax_kpi.set_title("KPI Overview", fontsize=11, color="#e8eaed",
                 fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 2 — Rating distribution (compact bar, log scale)
# ─────────────────────────────────────────────────────────────────────────────
rc2 = rating_counts
pct2 = rating_pct

bars2 = ax_dist.bar(
    rc2.index, rc2.values,
    color=[BAR_COLORS[r] for r in rc2.index],
    edgecolor="#0c0f14", linewidth=0.5, width=0.6, zorder=3,
)
ax_dist.set_yscale("log")

for bar, (rating, cnt) in zip(bars2, rc2.items()):
    ax_dist.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.55,
        f"{int(cnt):,}\n({pct2[rating]:.1f}%)",
        ha="center", va="bottom", fontsize=7, color="#e8eaed",
    )

ax_dist.set_xticks([1, 2, 3, 4, 5])
ax_dist.set_xticklabels(["1*", "2*", "3*", "4*", "5*"], fontsize=9)
ax_dist.set_ylabel("Count (log)", fontsize=9)
ax_dist.set_xlim(0.4, 5.6)
ax_dist.grid(axis="y", which="both", linestyle=":", linewidth=0.4, alpha=0.35)
ax_dist.spines[["top", "right"]].set_visible(False)
ax_dist.set_title("Rating Distribution", fontsize=11, color="#e8eaed",
                  fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 3 — CSAT by market (horizontal bars, best at top)
# ─────────────────────────────────────────────────────────────────────────────
ms3     = market_stats.sort_values("mean_csat", ascending=True)
y3      = range(len(ms3))
colors3 = list(reversed(bar_colors_market))   # best (top of sorted) → GREEN

bars3 = ax_market.barh(
    list(y3), ms3["mean_csat"],
    color=colors3, edgecolor="#0c0f14", linewidth=0.4,
    height=0.55, zorder=3,
)
ax_market.axvline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.0,
                  linestyle="--", alpha=0.6, zorder=2)
for bar, (_, row) in zip(bars3, ms3.iterrows()):
    ax_market.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{row['mean_csat']:.3f}",
        va="center", ha="left", fontsize=8, color="#e8eaed", fontweight="bold",
    )
ax_market.set_yticks(list(y3))
ax_market.set_yticklabels(ms3["market"], fontsize=9)
ax_market.set_xlim(
    ms3["mean_csat"].min() - 0.08,
    TARGET_CSAT + 0.06,
)
ax_market.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.35)
ax_market.spines[["top", "right"]].set_visible(False)
ax_market.set_title("Mean CSAT by Market", fontsize=11, color="#e8eaed",
                    fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 4 — Weekly trend line (no volume bars — keep it clean)
# ─────────────────────────────────────────────────────────────────────────────
ax_weekly.plot(
    weekly["week_number"], weekly["mean_csat"],
    color=ACCENT, linewidth=2.0,
    marker="o", markersize=5,
    markerfacecolor=ACCENT, markeredgecolor="#0c0f14", markeredgewidth=0.6,
    zorder=3,
)
ax_weekly.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=1.0,
                  linestyle="--", alpha=0.65, zorder=2)

for w, m in zip(weekly["week_number"], weekly["mean_csat"]):
    ax_weekly.text(w, m + 0.004, f"{m:.3f}",
                   ha="center", va="bottom", fontsize=7.5, color="#e8eaed")

ax_weekly.set_xticks(weekly["week_number"].tolist())
ax_weekly.set_xticklabels(
    [f"Wk{w}" for w in weekly["week_number"]], fontsize=8, rotation=30, ha="right"
)
ax_weekly.set_ylim(
    weekly["mean_csat"].min() - 0.04,
    weekly["mean_csat"].max() + 0.04,
)
ax_weekly.grid(linestyle=":", linewidth=0.4, alpha=0.35)
ax_weekly.spines[["top", "right"]].set_visible(False)
ax_weekly.set_title("Weekly CSAT Trend", fontsize=11, color="#e8eaed",
                    fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 5 — CSAT by hour (the cliff chart, compact)
# ─────────────────────────────────────────────────────────────────────────────
ax2h = ax_hour.twinx()

# Volume area (right axis, very muted)
ax2h.fill_between(
    hourly["hour_of_day"], hourly["count"],
    alpha=0.08, color=TEXT_MUTED, zorder=1,
)
ax2h.set_ylim(0, hourly["count"].max() * 3.5)
ax2h.tick_params(axis="y", colors=TEXT_MUTED, labelsize=7)
ax2h.spines[["top", "right"]].set_visible(False)

bar_h = ax_hour.bar(
    hourly["hour_of_day"], hourly["mean_csat"],
    color=[hour_bar_color(m) for m in hourly["mean_csat"]],
    edgecolor="#0c0f14", linewidth=0.3, width=0.75, zorder=3,
)
ax_hour.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=0.9,
                linestyle="--", alpha=0.6, zorder=2)

# Annotate only the two extreme drop hours
for dh in [2, 4]:
    row_dh = hourly[hourly["hour_of_day"] == dh]
    if row_dh.empty:
        continue
    dm = row_dh["mean_csat"].values[0]
    ax_hour.annotate(
        f"{dh:02d}h\n{dm:.1f}",
        xy=(dh, dm),
        xytext=(dh + 2.5, dm - 0.22),
        fontsize=7, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
        zorder=5,
    )

ax_hour.set_xticks(hourly["hour_of_day"].tolist())
ax_hour.set_xticklabels(
    [f"{h:02d}" for h in hourly["hour_of_day"]], fontsize=7, rotation=45, ha="right"
)
ax_hour.set_ylim(max(0, hourly["mean_csat"].min() - 0.3),
                 hourly["mean_csat"].max() + 0.1)
ax_hour.set_ylabel("Mean CSAT", fontsize=9)
ax_hour.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)
ax_hour.spines[["top", "right"]].set_visible(False)
ax_hour.set_zorder(ax2h.get_zorder() + 1)
ax_hour.patch.set_visible(False)
ax_hour.set_title("CSAT by Hour  (late-night cliff)", fontsize=11,
                  color="#e8eaed", fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 6 — Instructions impact (two bars + gap annotation)
# ─────────────────────────────────────────────────────────────────────────────
instr_colors6 = [BLUE, GREEN]
bars6 = ax_instr.bar(
    instr_agg["label"], instr_agg["mean_csat"],
    color=instr_colors6, edgecolor="#0c0f14", linewidth=0.6,
    width=0.42, zorder=3,
)
for bar6, (_, row6) in zip(bars6, instr_agg.iterrows()):
    h6 = bar6.get_height()
    ax_instr.text(
        bar6.get_x() + bar6.get_width() / 2,
        h6 + 0.002,
        f"{h6:.4f}",
        ha="center", va="bottom", fontsize=9,
        color="#e8eaed", fontweight="bold",
    )
    ax_instr.text(
        bar6.get_x() + bar6.get_width() / 2,
        h6 * 0.5,
        f"n={int(row6['count']):,}",
        ha="center", va="center", fontsize=8, color="#0c0f14",
    )

# Gap bracket
x0_6  = bars6[0].get_x() + bars6[0].get_width()
x1_6  = bars6[1].get_x()
yb_6  = max(instr_agg["mean_csat"]) + 0.010
ax_instr.annotate(
    "", xy=(x1_6, yb_6), xytext=(x0_6, yb_6),
    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4),
)
ax_instr.text(
    (x0_6 + x1_6) / 2, yb_6 + 0.003,
    f"Gap {instr_gap:+.4f}",
    ha="center", va="bottom", fontsize=9, color=ACCENT, fontweight="bold",
)

ax_instr.axhline(TARGET_CSAT, color=TEXT_MUTED, linewidth=0.9,
                 linestyle="--", alpha=0.6, zorder=2)
ax_instr.set_ylabel("Mean CSAT", fontsize=9)
y_lo6 = min(instr_agg["mean_csat"]) - 0.06
y_hi6 = max(instr_agg["mean_csat"]) + 0.06
ax_instr.set_ylim(y_lo6, y_hi6)
ax_instr.tick_params(axis="x", labelsize=9)
ax_instr.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)
ax_instr.spines[["top", "right"]].set_visible(False)
ax_instr.set_title("Instructions Impact on CSAT", fontsize=11,
                   color="#e8eaed", fontweight="bold", pad=8)

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "Veho CSAT Analysis  --  Executive Summary",
    fontsize=19, fontweight="bold", color="#e8eaed", y=0.965,
)

save_fig(fig, "19_executive_dashboard")
plt.close(fig)
