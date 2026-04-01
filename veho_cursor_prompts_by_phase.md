# Veho CSAT — Cursor Claude Code Prompts (Phase by Phase)
# Copy-paste each phase into Claude Code in Cursor sequentially.
# Each phase builds on the previous one.

# ============================================================
# HOW TO USE THIS FILE:
# 
# 1. Open Cursor
# 2. Create a new project folder with the CSV file inside
# 3. Open Claude Code (Cmd+L or the chat panel)
# 4. Copy-paste PHASE 1 prompt into Claude Code, let it generate & run
# 5. Review the output, then paste PHASE 2, etc.
# 6. Each phase will add to the same .py script
# ============================================================


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1: SETUP & LOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Create a Python script called `veho_csat_analysis.py`.

PHASE 1 — Setup and Data Loading:

1. Import: pandas, numpy, matplotlib.pyplot, seaborn, re, os, datetime
2. Configure a DARK chart theme using matplotlib rcParams:
   - Figure facecolor: '#0c0f14'
   - Axes facecolor: '#151920'
   - Text color: '#e8eaed'
   - Grid color: '#2a2f3a'
   - Default figure size: (12, 7)
   - DPI: 150
   - Font family: sans-serif
3. Define color constants:
   ACCENT = '#ff6b35'
   GREEN = '#34d399'
   RED = '#f87171'
   BLUE = '#60a5fa'
   YELLOW = '#fbbf24'
   PURPLE = '#a78bfa'
   TEXT_MUTED = '#8b8f98'
4. Create a `figures/` directory if it doesn't exist
5. Define a helper function `save_fig(fig, name)` that saves to figures/ at 300 DPI with tight bbox and dark facecolor, then calls plt.show()
6. Load `case_study_veho_Raw_Data.csv` into a DataFrame
7. Print: shape, columns, dtypes, null counts, unique values for market and last_event
8. Print 20 sample CSAT values to show how messy the field is

Comment style: block-level comments explaining the logic and reasoning, not line-by-line.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: DATA CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 2 — Data Cleaning:

The `csat` column is a free-text mess. Ratings are mixed with comments, emojis, and annotations.
Examples: "5", "4+", "5 thank you!", "(4) almost great 😊", "Thank you! 5", "5👍", "Actually, 1. It's not ours"

Build a function `extract_csat(val)` that uses regex to extract the numeric 1-5 rating:
- Priority order: direct match "1"-"5" → "4+" → starts with digit 1-5 → parens "(4)" → trailing digit → emoji-prefixed (first char is digit) → any digit 1-5 anywhere → NaN
- Return int or np.nan

Build `extract_comment(val)` that strips the numeric prefix and returns remaining text if >3 chars.

Apply both to create `csat_numeric` and `csat_comment` columns.

Parse datetimes:
- `route_start_time` and `updated_at` with format='ISO8601', utc=True
- Engineer: delivery_date, day_of_week, hour_of_day (from updated_at), week_number (ISO week), delivery_duration_hrs

Normalize city to uppercase. Normalize state to 2-letter codes (map "Texas"→"TX", "Colorado"→"CO", etc).

Create `valid` DataFrame = only rows where csat_numeric is not NaN.

Print cleaning summary: total rows, valid count, parse rate, date range, comment count, sample unparsed values.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3: CSAT DISTRIBUTION ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 3 — CSAT Distribution:

Chart 1 - `01_csat_distribution.png`:
Bar chart of rating counts (1 through 5). Use LOG scale on y-axis because 5-star dominates.
Color each bar: 1★=RED, 2★='#fb923c', 3★=YELLOW, 4★=BLUE, 5★=GREEN.
Annotate each bar with count + percentage. Add horizontal dashed line at 4.9 annotation.
Title: "CSAT Rating Distribution (Log Scale)"

Chart 2 - `02_csat_violin.png`:
Violin plot of csat_numeric showing the density shape. Overlay a box plot inside.
This visualizes the extreme left skew.

Print descriptive stats: mean, median, std, skewness (use scipy.stats.skew or compute manually), % per rating.
Print insight: "93.4% of ratings are 5-star. The gap to 4.9 is driven by {x}% of 1-star ratings."
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4: MARKET ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 4 — Market Analysis:

Chart 3 - `03_csat_by_market.png`:
Horizontal bar chart. Markets sorted by mean CSAT (best at top).
Vertical dashed line at 4.9 target. Color gradient from GREEN (best) to RED (worst).
Annotate each bar with exact mean value.

Chart 4 - `04_market_multi_metric.png`:
3 subplots side by side for each market: Mean CSAT, 5★ Rate %, Below-4★ Rate %.

Chart 5 - `05_misdelivery_by_market.png`:
Bar chart of misdelivery rate (%) by market. Annotate with count/total.

Print a formatted table: market | count | mean_csat | 5star_pct | below4_pct | misdelivery_rate.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 5: TEMPORAL TRENDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 5 — Temporal Trends:

Chart 6 - `06_csat_by_week.png`:
Dual-axis: line for mean CSAT (left axis), bars for delivery volume (right axis).
Dashed line at 4.9 target. Show upward trend Week 11→15.

Chart 7 - `07_csat_market_weekly.png`:
Multi-line chart, one line per market, x=week, y=mean CSAT.
Use distinct colors per market. Legend at bottom.

Chart 8 - `08_csat_by_dow.png`:
Bar chart Monday→Sunday. Highlight Thursday (best) in GREEN and Saturday (worst) in YELLOW.

Chart 9 - `09_csat_by_hour.png`:
THIS IS THE MOST IMPORTANT CHART. Bars showing mean CSAT by delivery hour (from updated_at).
Color bars by threshold: GREEN if >=4.8, YELLOW if >=4.5, orange if >=4.0, RED if <4.0.
Overlay a line showing delivery volume per hour (right axis).
Add text annotations at the extreme drop points (2AM, 4AM).
This clearly shows the late-night delivery cliff.

Print weekly trend table and the critical time finding.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 6: DRIVER PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 6 — Driver Analysis:

Compute per-driver stats for drivers with 10+ deliveries (qualified).

Chart 10 - `10_driver_csat_histogram.png`:
Histogram of driver mean CSAT values. Add vertical lines for overall mean and 4.9 target.
Shade the area below 4.5 in red to highlight problematic drivers.

Chart 11 - `11_bottom_drivers.png`:
Horizontal bars for bottom 15 drivers. Color-code by market.
Annotate with delivery count.

Chart 12 - `12_driver_scatter.png`:
Scatter: x=delivery count, y=mean CSAT, color=market, alpha=0.5.
Shows that low-volume drivers have higher variance.

Print: total drivers, qualified count, below 4.5 count, below 4.0 count, bottom 10 table.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 7: OPERATIONS DEEP-DIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 7 — Operations:

Chart 13 - `13_duration_vs_csat.png`:
Bar chart: mean delivery duration (hours from route_start to updated_at) grouped by CSAT rating 1-5.
Filter out duration < 0 or > 24 hours. Shows longer deliveries = lower CSAT.

Chart 14 - `14_instructions_impact.png`:
Side-by-side bars: "With Instructions" vs "Without Instructions" showing mean CSAT.
Annotate the 0.134 gap. This is a key finding.

Chart 15 - `15_misdelivery_impact.png`:
Two bars: delivered (4.84 CSAT) vs misdelivered (1.18 CSAT). Dramatic visual.

Chart 16 - `16_route_size_vs_csat.png`:
Compute mean CSAT by route size buckets: 1-3, 4-6, 7-9, 10-12, 13+ packages.
Bar chart showing if larger routes → lower CSAT.

Print insights for each finding.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 8: CUSTOMER VOICE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 8 — Customer Voice Analysis:

From rows where csat_numeric <= 3 AND csat_comment is not null:

Chart 17 - `17_low_csat_keywords.png`:
Tokenize comments (lowercase, split on whitespace/punctuation). Remove stopwords manually:
["the","a","an","is","was","it","to","and","of","in","for","my","i","we","not","that","this","but","with","have","had","are","be","on","at","from","or","so","if","just","been","has","very","can","do","did","no","me","our","you","your","they","them","would","will","should","could","its","im","dont","ive","doesnt","were","about","all","one","up","out","get","got","also","than","when","more"]
Count word frequency. Horizontal bar chart of top 20 words.

Chart 18 - `18_comment_themes.png`:
Classify each low-CSAT comment into themes using keyword matching:
- "wrong" OR "address" OR "house" OR "misdeliver" OR "not my" OR "not mine" → "Wrong Address"
- "door" OR "instruction" OR "direction" OR "left" OR "place" OR "front" OR "behind" OR "leave" → "Instructions Not Followed"
- "late" OR "time" OR "early" OR "hours" OR "long" OR "wait" → "Late Delivery"  
- "rude" OR "driver" OR "behavior" OR "attitude" OR "shouted" OR "expletive" → "Driver Behavior"
- "cold" OR "warm" OR "damaged" OR "leak" OR "broken" OR "condition" → "Package Condition"
- "stolen" OR "missing" OR "never" OR "no delivery" OR "not delivered" → "Missing/Stolen"
- else → "Other"
Note: a comment can match multiple themes. Count first match only.
Donut chart showing distribution.

Print 15 sample low-CSAT comments. Print theme counts.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 9: EXECUTIVE DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 9 — Summary Dashboard:

Chart 19 - `19_executive_dashboard.png`:
Create a single large figure (18 x 22 inches) with 6 subplots in a 3x2 grid:

Subplot 1 (top-left): TEXT-ONLY KPI panel. Use ax.text() to display:
  "Overall CSAT: 4.836 | Target: 4.9"
  "5★ Rate: 93.4% | Below 4★: 4.4%"
  "Misdelivery Rate: 0.11% | Drivers: 1,940"
  Make this look like a KPI card with large bold text.

Subplot 2 (top-right): CSAT distribution bar chart (same as Chart 1 but compact)
Subplot 3 (mid-left): CSAT by market horizontal bars
Subplot 4 (mid-right): CSAT by week trend line
Subplot 5 (bottom-left): CSAT by delivery hour (the cliff chart)
Subplot 6 (bottom-right): Instructions impact comparison

Main title: "Veho CSAT Analysis — Executive Summary"
This is the one chart to screen-share in 2 minutes.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 10: FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Continue the script. PHASE 10 — Final Output:

Print this formatted summary to console (replace placeholders with actual computed values):

============================================================
VEHO CSAT ANALYSIS — KEY FINDINGS SUMMARY
============================================================

DATASET: {total} deliveries | {markets} markets | {date_min} - {date_max}
OVERALL CSAT: {mean:.3f} (target: 4.9, gap: {gap:.3f})
5-STAR RATE: {pct5:.1f}% | BELOW 4-STAR: {pctbelow4:.1f}%

TOP INSIGHTS:
1. CSAT is a long-tail problem — 93% perfect, 4% terrible
2. Late-night deliveries (after 10 PM) → CSAT drops below 4.5
3. Misdeliveries are rare (0.11%) but catastrophic (avg CSAT: 1.18)
4. Instructions not followed is the #1 addressable root cause
5. 8 drivers below 4.0 CSAT — highest-leverage coaching targets
6. Philadelphia & Baltimore consistently underperform

RECOMMENDED ACTIONS:
[Immediate] Coach/pause bottom 8 drivers. Cap late-night deliveries.
[Short-term] CSAT alerting per market. Instructions compliance audit.
[Medium-term] Predictive CSAT model. Driver incentive A/B tests.
[Strategic] NLP-based instructions compliance. CSAT in marketplace ranking.

All {n} figures saved to figures/ directory.
============================================================

Also print: "Script completed successfully in {elapsed:.1f} seconds"
(Track elapsed time from the start of the script using time.time())
"""
