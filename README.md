# freight-rate-prediction

## Run
```bash
python -m pip install -r requirements.txt
# put the 4 CSVs in ./data/ (train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv)
python train.py                     # validates, trains, writes validation_predictions.csv + december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```
`python make_december_chart.py` is a fallback chart if score.py is unavailable.

## Files
- `features.py` – feature engineering (no city IDs; geography, weight cleaning, calendar)
- `model.py` – outlier removal + 3-member LightGBM ensemble
- `train.py` – time-based and unseen-city validation, final fit, predictions
- `experiments/` – EDA / ablation scripts that justified the design
- `outputs/validation_report.json` – validation metrics
