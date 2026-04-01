# Veho CSAT Analysis

Exploratory analysis of customer satisfaction data from Veho last-mile delivery operations. The script processes raw case-study data, cleans it, and produces 19 publication-quality charts across 9 analytical phases.

## Quick start

```bash
pip install pandas numpy matplotlib seaborn scipy
python veho_csat_analysis.py
```

All figures are written to `figures/`.

## Data

| File | Description |
|------|-------------|
| `case_study_veho_Raw_Data.csv` | Raw export — 76,497 rows, 15 columns |

Key columns: `csat`, `driver_id`, `market`, `route_id`, `package_id`, `last_event`, `route_start_time`, `updated_at`, `instructions`, `city`, `state`.

## Key findings

| Finding | Value |
|---------|-------|
| Overall mean CSAT | **4.832** (target: 4.9) |
| 5-star rate | 93.3% |
| Skewness | −4.56 (extreme left skew) |
| Best market | Dallas (4.862) |
| Worst market | Philadelphia (4.813) |
| Misdelivery CSAT | **1.20** vs 4.84 delivered (−3.64 pts) |
| Late-night cliff | 02:00–04:00 CSAT drops to **2.76** vs 4.87 daytime |
| Top complaint theme | Instructions Not Followed (35.8% of voiced complaints) |
| Bottom drivers (≥10 del.) | 72 qualified drivers below 4.5 CSAT (5.1%) |

## Phases & charts

### Phase 1 — Data Inspection
Prints shape, dtypes, null counts, and cardinality for `market` and `last_event`.

### Phase 2 — Data Cleaning
- `extract_csat()` — priority-ladder regex parser; recovers ratings from free text, emoji prefixes, and annotated strings. Parse rate: **100%**.
- `extract_comment()` — strips the rating token and returns leftover comment text.
- ISO8601 datetime parsing with UTC normalisation.
- Derived features: `delivery_date`, `day_of_week`, `hour_of_day`, `week_number`, `delivery_duration_hrs`.
- State normalisation to 2-letter codes.

### Phase 3 — CSAT Distribution

| Chart | File | Description |
|-------|------|-------------|
| 1 | `01_csat_distribution.png` | Bar chart of rating counts, log y-axis, tier colours, count+% annotations |
| 2 | `02_csat_violin.png` | Violin + box overlay showing the extreme left skew |

### Phase 4 — Market Analysis

| Chart | File | Description |
|-------|------|-------------|
| 3 | `03_csat_by_market.png` | Horizontal bars, GREEN→RED gradient, 4.9 target line |
| 4 | `04_market_multi_metric.png` | 3-panel: mean CSAT / 5-star rate / below-4-star rate |
| 5 | `05_misdelivery_by_market.png` | Misdelivery rate by market, count/total annotated |

### Phase 5 — Temporal Trends

| Chart | File | Description |
|-------|------|-------------|
| 6 | `06_csat_by_week.png` | Dual-axis: CSAT line + volume bars, Wk 11→14 uptrend arrow |
| 7 | `07_csat_market_weekly.png` | Multi-line weekly CSAT, one line per market |
| 8 | `08_csat_by_dow.png` | Mon→Sun bars, best day highlighted GREEN, worst YELLOW |
| 9 | `09_csat_by_hour.png` | **Headline chart.** Threshold-coloured bars, volume overlay, 02h/04h cliff callouts |

### Phase 6 — Driver Analysis

Qualified = drivers with ≥ 10 deliveries (1,415 of 1,940 total).

| Chart | File | Description |
|-------|------|-------------|
| 10 | `10_driver_csat_histogram.png` | Distribution of driver mean CSAT, red shading below 4.5 |
| 11 | `11_bottom_drivers.png` | Bottom 15 drivers, market-coloured, delivery count annotated |
| 12 | `12_driver_scatter.png` | Scatter: delivery count vs mean CSAT — variance shrinks with volume |

### Phase 7 — Operations

| Chart | File | Description |
|-------|------|-------------|
| 13 | `13_duration_vs_csat.png` | Mean delivery duration by star rating (1-star: +0.53 hrs vs 5-star) |
| 14 | `14_instructions_impact.png` | With vs without instructions — gap annotation |
| 15 | `15_misdelivery_impact.png` | Delivered vs misdelivered — 3.64-point CSAT collapse |
| 16 | `16_route_size_vs_csat.png` | Mean CSAT by route-size bucket (1-3, 4-6, 7-9, 10-12, 13+) |

### Phase 8 — Customer Voice

Source: 1,045 comments where `csat_numeric <= 3` and `csat_comment` is not null.

| Chart | File | Description |
|-------|------|-------------|
| 17 | `17_low_csat_keywords.png` | Top 20 keywords after stopword removal, RED gradient |
| 18 | `18_comment_themes.png` | Donut chart of 7 complaint themes via keyword classification |

**Theme breakdown:**

| Theme | % |
|-------|---|
| Instructions Not Followed | 35.8% |
| Other | 32.7% |
| Wrong Address | 15.7% |
| Late Delivery | 9.3% |
| Package Condition | 2.7% |
| Missing/Stolen | 2.5% |
| Driver Behavior | 1.3% |

### Phase 9 — Executive Dashboard

| Chart | File | Description |
|-------|------|-------------|
| 19 | `19_executive_dashboard.png` | 3×2 grid: KPI card, distribution, market bars, weekly trend, hour cliff, instructions impact |

## Color palette

| Name | Hex | Used for |
|------|-----|----------|
| ACCENT | `#ff6b35` | Primary line/highlight |
| GREEN | `#34d399` | Best / healthy (≥4.8) |
| BLUE | `#60a5fa` | Neutral / volume |
| YELLOW | `#fbbf24` | Caution (≥4.5) |
| RED | `#f87171` | Critical / worst |
| PURPLE | `#a78bfa` | Philadelphia market |
| TEXT_MUTED | `#8b8f98` | Reference lines / secondary labels |

Background: `#0c0f14` (figure) / `#151920` (axes).
