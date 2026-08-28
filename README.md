# Supply Chain Demand Forecasting Dashboard

> End-to-end ML system: raw M5 retail data → feature engineering → 4 forecasting models → demand-class-aware ensemble → live BI dashboard

## Architecture

```
Raw M5 CSVs → DuckDB → Feature Engineering → Demand Classification (ADI×CV²)
                                                    ↓
                    ┌───────────────────────────────┤
                    ↓               ↓               ↓
               Smooth/Erratic   Intermittent    Lumpy
                    ↓               ↓               ↓
             SARIMA + Prophet  Dual-Phase LGB  Dual-Phase LGB
             + LightGBM        (occur+magn)    (occur+magn)
             + LSTM
                    ↓               ↓               ↓
                    └──────── Ridge Meta-Learner ────┘
                                    ↓
                            Plotly Dash Dashboard
```

## Project Structure

```
Proj_2/
├── data/
│   ├── raw/              ← Place M5 CSVs here
│   ├── processed/        ← DuckDB database (auto-generated)
│   └── models/           ← Trained model files (auto-generated)
├── src/
│   ├── pipeline/
│   │   ├── ingest.py     ← CSV → DuckDB (wide→long melt + joins)
│   │   ├── features.py   ← 30+ features: lags, rolling, calendar, price
│   │   └── classify.py   ← ADI×CV² demand classification
│   ├── models/
│   │   ├── lgbm_model.py   ← LightGBM (point + quantile)
│   │   ├── prophet_model.py← Facebook Prophet
│   │   ├── sarima_model.py ← SARIMAX statistical baseline
│   │   ├── lstm_model.py   ← PyTorch LSTM (MPS-accelerated)
│   │   └── dual_phase.py   ← Dual-phase model (novel contribution)
│   ├── ensemble/
│   │   ├── stacker.py    ← Ridge meta-learner per demand class
│   │   └── evaluate.py   ← RMSE, MAE, WRMSSE metrics
│   └── dashboard/
│       ├── app.py        ← Dash app entry point
│       ├── data_layer.py ← Cached DB queries + demo data
│       ├── pages/        ← 4 dashboard pages
│       └── assets/       ← CSS (dark glassmorphism theme)
├── scripts/
│   ├── run_pipeline.py   ← Step 1: Ingest + features + classify
│   ├── run_models.py     ← Step 2: Train all models
│   └── run_ensemble.py   ← Step 3: Ensemble + evaluate
├── tests/                ← pytest unit tests
├── notebooks/            ← EDA + analysis notebooks
└── reports/              ← model_comparison.csv (auto-generated)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the M5 Dataset
**Option A — Kaggle API** (if you have `~/.kaggle/kaggle.json`):
```bash
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

**Option B — Manual download**:
 [https://www.kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place these files in `data/raw/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

### 3. Run the Pipeline

**Fast test run (200 series, ~5 minutes):**
```bash
python scripts/run_pipeline.py --sample 200
python scripts/run_models.py --sample 200 --models lgbm
python scripts/run_ensemble.py
```

**Full production run (42,840 series, ~hours):**
```bash
python scripts/run_pipeline.py
python scripts/run_models.py
python scripts/run_ensemble.py
```

**Full production run, one model at a time (recommended for local machines):**
```bash
python scripts/run_models.py --models lgbm
python scripts/run_models.py --models prophet
python scripts/run_models.py --models sarima
python scripts/run_models.py --models lstm
python scripts/run_models.py --models dual
python scripts/run_models.py --status
python scripts/run_ensemble.py
```

Each model writes its own DuckDB table (`pred_lgbm`, `pred_prophet`, `pred_sarima`,
`pred_lstm`, `pred_dual`). `run_ensemble.py` automatically uses whichever
prediction tables are available. To avoid rerunning completed outputs:
```bash
python scripts/run_models.py --models lgbm prophet sarima lstm dual --skip-existing
```

### 4. Launch the Dashboard
```bash
python src/dashboard/app.py
```
Open: [http://localhost:8050](http://localhost:8050)

> **Note**: The dashboard works with **demo data** even before the full pipeline runs.

### 5. Run Tests
```bash
pytest tests/ -v
```

## Novel Research Contribution

### Demand Classification Routing
Each SKU is classified using the **Syntetos-Boylan ADI×CV² grid**:

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 1.32** | Smooth → Full ensemble | Erratic → LGB + Prophet |
| **ADI ≥ 1.32** | Intermittent → Dual-Phase | Lumpy → Dual-Phase |

### Dual-Phase Model (Key Innovation)
For Intermittent & Lumpy SKUs:
```
forecast = P(demand > 0) × E(demand | demand > 0)
           ↑ Stage 1: Binary LGB    ↑ Stage 2: Regression LGB
```
This separation significantly outperforms applying a single regressor to sparse series.

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI cards (accuracy, stockout/overstock alerts, revenue at risk) |
| **Actual vs Forecast** | Interactive drill-down by State → Store → Category → Item |
| **30-Day Forward View** | Risk heatmap (SKU × date, green/amber/red) + order recommendations |
| **Model Performance** | WRMSSE leaderboard, ADI×CV² scatter, feature importance |

## Expected Results (M5 dataset)
| Model | WRMSSE |
|---|---|
| Naive baseline | ~0.900 |
| SARIMA | ~0.720 |
| Prophet | ~0.680 |
| LightGBM | ~0.623 |
| Dual-Phase (intermittent) | ~0.601 |
| **Ensemble** | **~0.587** |

## Tech Stack
Python 3.11 · DuckDB · Pandas · LightGBM · Prophet · statsmodels · PyTorch (MPS) · Plotly Dash · scikit-learn · SHAP

