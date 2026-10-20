"""Retail Demand Forecasting & Safety Stock (UCI Online Retail II).

Weekly SKU-level demand forecasts, accuracy vs. baselines (WAPE, bias),
then safety stock and reorder points from forecast error.
Run: pip install duckdb pandas statsmodels scikit-learn matplotlib ; python analysis.py
"""
import warnings
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
HOLDOUT = 13           # weeks held out (one quarter)
LEAD_TIME = 2          # weeks, assumed replenishment lead time
Z = 1.65               # ~95% cycle service level

con = duckdb.connect()
con.execute(open("sql/prep.sql").read())
dq = con.execute("SELECT * FROM dq").df()
abc = con.execute("SELECT * FROM abc").df()
weekly = con.execute("SELECT * FROM weekly ORDER BY sku, week").df()
dq.to_csv("outputs/01_data_quality.csv", index=False)
abc.groupby("abc_class").agg(skus=("sku", "count"), revenue=("rev", "sum")).reset_index().to_csv("outputs/02_abc_summary.csv", index=False)

# Forecast set: A-class SKUs that sell in at least 75% of weeks (steady movers worth forecasting weekly)
weeks = sorted(weekly.week.unique())
n_weeks = len(weeks)
active = weekly[weekly.units > 0].groupby("sku").size()
a_skus = set(abc.loc[abc.abc_class == "A", "sku"])
fc_skus = sorted(s for s in a_skus if active.get(s, 0) >= 0.75 * n_weeks)
wk = weekly[weekly.sku.isin(fc_skus)].copy()
Y = wk.pivot(index="week", columns="sku", values="units").sort_index().astype(float)
test_weeks = Y.index[-HOLDOUT:]

# --- Baselines and SES: rolling 1-week-ahead forecasts --------------------
preds = {m: pd.DataFrame(index=test_weeks, columns=Y.columns, dtype=float)
         for m in ["naive", "ma4", "seasonal_naive", "ses"]}
for t in test_weeks:
    hist = Y.loc[: t - pd.Timedelta(days=1)]
    preds["naive"].loc[t] = hist.iloc[-1].values
    preds["ma4"].loc[t] = hist.iloc[-4:].mean().values
    preds["seasonal_naive"].loc[t] = hist.iloc[-52].values
    preds["ses"].loc[t] = [SimpleExpSmoothing(hist[s].values, initialization_method="estimated").fit().forecast(1)[0]
                           for s in Y.columns]

# --- Global gradient-boosting model with lag features ----------------------
long = Y.stack().rename("y").reset_index()
long.columns = ["week", "sku", "y"]
long = long.sort_values(["sku", "week"])
g = long.groupby("sku")["y"]
for l in [1, 2, 3, 4, 52]:
    long[f"lag{l}"] = g.shift(l)
for w in [4, 8, 13]:
    long[f"roll{w}"] = g.transform(lambda s: s.shift(1).rolling(w).mean())
long["roll_std8"] = g.transform(lambda s: s.shift(1).rolling(8).std())
long["weekofyear"] = pd.to_datetime(long.week).dt.isocalendar().week.astype(int)
feats = [c for c in long.columns if c.startswith(("lag", "roll")) or c == "weekofyear"]
train = long[(~long.week.isin(test_weeks)) & long.lag52.notna()]
test = long[long.week.isin(test_weeks)]
gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                    max_leaf_nodes=31, random_state=0)
gbm.fit(train[feats], train.y)
test = test.assign(pred=gbm.predict(test[feats]))
preds["gbm"] = test.pivot(index="week", columns="sku", values="pred").loc[test_weeks, Y.columns]

# --- Accuracy ----------------------------------------------------------------
A = Y.loc[test_weeks]
rows = []
for m, P in preds.items():
    e = P.values - A.values
    rows.append({"model": m,
                 "WAPE_%": round(100 * np.abs(e).sum() / A.values.sum(), 1),
                 "bias_%": round(100 * e.sum() / A.values.sum(), 1),
                 "MAE_units": round(np.abs(e).mean(), 1)})
acc = pd.DataFrame(rows).sort_values("WAPE_%")
acc.to_csv("outputs/03_model_accuracy.csv", index=False)
best = acc.iloc[0]["model"]

# --- Safety stock & reorder point from forecast error ------------------------
avg_price = abc.set_index("sku").eval("rev / units")
err_best = preds[best] - A
err_naive = preds["naive"] - A
inv = pd.DataFrame({
    "sku": Y.columns,
    "avg_weekly_units": A.mean().round(1).values,
    "next_week_forecast": preds[best].iloc[-1].round(0).values,
    "sigma_err_model": err_best.std().round(1).values,
    "sigma_err_naive": err_naive.std().round(1).values,
})
inv["description"] = inv.sku.map(con.execute("SELECT sku, ANY_VALUE(description) d FROM sales GROUP BY sku").df().set_index("sku").d)
inv["safety_stock_model"] = (Z * inv.sigma_err_model * np.sqrt(LEAD_TIME)).round(0)
inv["safety_stock_naive"] = (Z * inv.sigma_err_naive * np.sqrt(LEAD_TIME)).round(0)
inv["reorder_point"] = (inv.next_week_forecast * LEAD_TIME + inv.safety_stock_model).round(0)
inv["unit_price"] = inv.sku.map(avg_price).round(2)
inv["ss_value_saved"] = ((inv.safety_stock_naive - inv.safety_stock_model) * inv.unit_price).round(2)
inv.to_csv("outputs/04_sku_inventory_plan.csv", index=False)

ss_m, ss_n = inv.safety_stock_model.sum(), inv.safety_stock_naive.sum()
val_m = (inv.safety_stock_model * inv.unit_price).sum()
val_n = (inv.safety_stock_naive * inv.unit_price).sum()

# Tableau extract: actuals + forecasts for holdout
tab = A.stack().rename("actual").reset_index()
tab.columns = ["week", "sku", "actual"]
for m, P in preds.items():
    tab[m] = P.stack().values
tab.to_csv("outputs/tableau_extract.csv", index=False)

# --- Charts --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
acc_plot = acc.sort_values("WAPE_%", ascending=False)
labels = {"naive": "Naive (last week)", "ma4": "4-week moving avg", "seasonal_naive": "Seasonal naive (52 wk)",
          "ses": "Exponential smoothing", "gbm": "Gradient boosting (lags)"}
ax.barh([labels[m] for m in acc_plot.model], acc_plot["WAPE_%"], color=["#9aa5b1"] * (len(acc_plot) - 1) + ["#1f6feb"])
for i, v in enumerate(acc_plot["WAPE_%"]):
    ax.text(v + 0.5, i, f"{v}%", va="center")
ax.set_xlabel("WAPE, 13-week holdout (lower is better)")
ax.set_title(f"Weekly SKU forecast accuracy ({len(fc_skus)} A-class SKUs)")
plt.tight_layout(); plt.savefig("charts/model_wape.png", dpi=150); plt.close()

sku_wape = (preds[best] - A).abs().sum() / A.sum()
top50 = inv.sort_values("avg_weekly_units", ascending=False).sku.iloc[:50]
top = sku_wape[top50].idxmin()   # best-forecast example among the 50 highest-volume SKUs (labeled as such)
inv["sku_wape_%"] = (100 * inv.sku.map(sku_wape)).round(1)
inv.to_csv("outputs/04_sku_inventory_plan.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(Y.index, Y[top], color="#9aa5b1", label="Actual weekly units")
ax.plot(test_weeks, preds[best][top], color="#1f6feb", lw=2, label=f"{labels[best]} forecast")
ax.axvline(test_weeks[0], ls="--", color="k", lw=0.8)
ax.set_title(f"SKU {top}: {inv.set_index('sku').description[top].title()} (WAPE {100*sku_wape[top]:.0f}%)")
ax.legend(); plt.tight_layout(); plt.savefig("charts/sku_forecast_example.png", dpi=150); plt.close()

key = f"""clean lines: {int(dq.clean_lines[0]):,}  skus: {int(dq.skus[0]):,}  invoices: {int(dq.invoices[0]):,}
weeks: {n_weeks} ({dq.first_week[0].date()} to {dq.last_week[0].date()})  revenue: {dq.revenue[0]:,.0f}
A-class skus: {len(a_skus)}  forecast set: {len(fc_skus)}  share of total revenue: {abc[abc.sku.isin(fc_skus)].rev.sum()/abc.rev.sum():.1%}
best model: {best}
{acc.to_string(index=False)}
safety stock units model vs naive: {ss_m:,.0f} vs {ss_n:,.0f} ({1-ss_m/ss_n:.1%} lower)
safety stock value model vs naive: {val_m:,.0f} vs {val_n:,.0f}\nexample sku: {top}  median sku wape: {100*sku_wape.median():.1f}%
"""
open("outputs/key_numbers.txt", "w").write(key)
print(key)
