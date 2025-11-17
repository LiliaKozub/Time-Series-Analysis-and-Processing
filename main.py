import os
import pandas as pd

from src import (
    data_loader,
    split,
    ses_manual,
    models,
    evaluation,
    visualization,
    utils
)

# ===========================
#   CONFIG
# ===========================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_CUSTOM_DATA = False
CUSTOM_CSV_PATH = "data/my_series.csv"
CUSTOM_DATE_COL = "date"
CUSTOM_VALUE_COL = "value"

# ===========================
#   LOAD DATA
# ===========================
ts = (
    data_loader.load_from_csv(CUSTOM_CSV_PATH, CUSTOM_DATE_COL, CUSTOM_VALUE_COL)
    if USE_CUSTOM_DATA else
    data_loader.load_airpassengers()
)

train, val, test = split.train_val_test_split(ts)

# ===========================
#   SES MANUAL (Grid Search)
# ===========================
best_alpha, best_rmse, ses_forecast_val = ses_manual.grid_search_ses(train, val)

ses_forecast_test = ses_manual.ses_forecast_last(
    pd.concat([train, val]),
    best_alpha,
    len(test)
)

# ===========================
#   STATSMODELS MODELS
# ===========================
sm_results_val = models.fit_statsmodels_models(train, val)

sm_results_test = {}
for name, info in sm_results_val.items():
    fit = info['fit']
    if fit is not None:
        sm_results_test[name] = {"forecast": fit.forecast(len(test))}
    else:
        sm_results_test[name] = {"forecast": None}

# ===========================
#   EVALUATION
# ===========================

val_scores_sm = evaluation.evaluate_models(val, sm_results_val)
val_scores_ses = evaluation.evaluate_models(val, {"SES_manual": {"forecast": ses_forecast_val}})

val_scores = {**val_scores_sm, **val_scores_ses}

all_forecasts_test = {**sm_results_test, "SES_manual": {"forecast": ses_forecast_test}}
test_scores = evaluation.evaluate_models(test, all_forecasts_test)

# ===========================
#   VISUALIZATION
# ===========================
visualization.plot_series_with_forecasts(
    train,
    val,
    test,
    all_forecasts_test,
    title='Series with forecasts',
    fname=os.path.join(OUTPUT_DIR, 'series_forecasts.png')
)

visualization.decompose_series(
    ts,
    fname=os.path.join(OUTPUT_DIR, 'decomposition.png')
)

# ===========================
#   SAVE FORECASTS
# ===========================
utils.save_all_forecasts(
    all_forecasts_test,
    os.path.join(OUTPUT_DIR, "forecasts_all_models.csv")
)

for name, info in all_forecasts_test.items():
    forecast = info.get("forecast")
    if forecast is not None:
        utils.save_forecast_single(
            name,
            forecast,
            OUTPUT_DIR
        )

# ===========================
#   SAVE METRICS
# ===========================
utils.save_metrics(val_scores, os.path.join(OUTPUT_DIR, "val_scores.csv"))
utils.save_metrics(test_scores, os.path.join(OUTPUT_DIR, "test_scores.csv"))

# ===========================
#   LOG OUTPUT
# ===========================
print("Validation scores (Statsmodels):", val_scores_sm)
print("Validation score (SES_manual):", val_scores_ses)
print("Test scores:", test_scores)
print("Done! Results saved to:", OUTPUT_DIR)
