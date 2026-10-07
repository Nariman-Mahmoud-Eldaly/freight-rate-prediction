"""Fallback chart script (the official score.py was not in the upload bundle). Run the official
`python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv`
to produce the official scorer_results/candidate_december.png."""
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
df = pd.read_csv("december_chart_inputs.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(df.date, df.predicted_rate, marker="o", ms=4, lw=1.8, color="#0b3d4d")
ax.set_title("December predicted rate: Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb")
ax.set_ylabel("Predicted rate ($)"); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
Path("scorer_results").mkdir(exist_ok=True); fig.savefig("scorer_results/candidate_december.png", dpi=150)
