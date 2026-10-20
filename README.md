# Retail Demand Forecasting & Safety Stock

**Question:** How accurately can weekly SKU demand be forecast, and how much safety stock does a better forecast save compared with a simple "same as last week" rule?

**Data:** UCI Machine Learning Repository, *Online Retail II* (public): 1,067,371 transaction lines from a UK online gift wholesaler, Dec 2009 to Dec 2011. Download: https://archive.ics.uci.edu/dataset/502/online+retail+ii

**Tools:** SQL (DuckDB), Python (pandas, scikit-learn, statsmodels, matplotlib), Tableau-ready extract.

## Method
1. **Clean** (`sql/prep.sql`): remove cancellations, returns, zero/negative prices, and non-product codes (postage, fees, manual adjustments). This leaves 1,019,490 lines, 4,706 SKUs, 39,007 invoices, and 103 full weeks.
2. **ABC classification** by revenue: 1,029 A-class SKUs make up 80% of revenue.
3. **Forecast set:** the 583 A-class SKUs that sell in at least 75% of weeks (54.3% of total revenue), with weekly demand zero-filled.
4. **Rolling 1-week-ahead forecasts** over a 13-week holdout (Sep to Nov 2011), comparing five models: naive (last week), 4-week moving average, seasonal naive (same week last year), simple exponential smoothing, and a global gradient-boosting model. The boosting model uses lags 1 to 4 and 52, rolling means and volatility, and week of year.
5. **Accuracy:** WAPE (total absolute error divided by total units), bias, and MAE.
6. **Inventory policy:** safety stock = z × σ(forecast error) × √(lead time), with z = 1.65 (about 95% cycle service) and an assumed 2-week lead time. Reorder point = forecast × lead time + safety stock.

## Results
| Model | WAPE | Bias |
|---|---|---|
| Gradient boosting (lags) | **59.8%** | −7.1% |
| Exponential smoothing | 61.0% | −4.0% |
| 4-week moving average | 61.8% | −7.2% |
| Naive (last week) | 73.3% | −3.2% |
| Seasonal naive (52 wk) | 98.7% | +33.1% |

- The best model cuts WAPE by **13.5 points** compared with the naive forecast. Median per-SKU WAPE is 67.0%.
- Its smaller errors mean **23.3% less safety stock** at the same service level: 110,983 vs. 144,648 units, or about £219K vs. £284K in stock value at average selling price.
- Seasonal naive does worst because demand grew year over year (+33% bias). Last year's pattern alone isn't enough.
- All models slightly under-forecast the Q4 ramp-up. The next improvement would be a holiday/seasonality feature or a review of order sizes by customer.

## Limitations
This is B2B wholesale demand, so orders are lumpy and large and weekly WAPE runs higher than it would for grocery store-level sales. The lead time and service level are assumptions. Safety stock values use average selling price, not cost. The model is evaluated on one 13-week window.

## Files
- `sql/prep.sql`: cleaning, data-quality profile, ABC classes, zero-filled weekly demand
- `analysis.py`: models, accuracy, safety stock and reorder points, charts
- `outputs/`: accuracy table, SKU inventory plan (forecast, σ, safety stock, reorder point, per-SKU WAPE), `tableau_extract.csv`
- `charts/`: model comparison and an example SKU forecast

Run: `pip install duckdb pandas scikit-learn statsmodels matplotlib openpyxl`. Then save the dataset as `data/online_retail_ii.csv` (both Excel sheets combined) and run `python analysis.py`.
