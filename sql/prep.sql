-- Retail Demand Forecasting: data prep (DuckDB)
-- Source: UCI Online Retail II (online_retail_ii.csv), Dec 2009 - Dec 2011

-- 1. Clean transaction lines: drop cancellations, returns/adjustments, non-product codes, bad prices
CREATE OR REPLACE TABLE sales AS
SELECT
    Invoice,
    UPPER(TRIM(StockCode))                 AS sku,
    TRIM(Description)                      AS description,
    Quantity                               AS qty,
    Price                                  AS unit_price,
    Quantity * Price                       AS revenue,
    CAST(InvoiceDate AS TIMESTAMP)         AS ts,
    date_trunc('week', CAST(InvoiceDate AS TIMESTAMP)) AS week
FROM read_csv_auto('data/online_retail_ii.csv', all_varchar=false)
WHERE Invoice NOT LIKE 'C%'
  AND Quantity > 0
  AND Price > 0
  AND regexp_matches(UPPER(StockCode), '^[0-9]{5}')          -- product SKUs only (drops POST, DOT, M, BANK CHARGES, etc.)
  AND CAST(InvoiceDate AS TIMESTAMP) < TIMESTAMP '2011-12-05'; -- last full week

-- 2. Data-quality profile
CREATE OR REPLACE TABLE dq AS
SELECT COUNT(*) AS clean_lines, COUNT(DISTINCT sku) AS skus, COUNT(DISTINCT Invoice) AS invoices,
       MIN(week) AS first_week, MAX(week) AS last_week, ROUND(SUM(revenue),0) AS revenue
FROM sales;

-- 3. ABC classification by revenue (A = top 80% of revenue)
CREATE OR REPLACE TABLE abc AS
WITH r AS (SELECT sku, SUM(revenue) AS rev, SUM(qty) AS units FROM sales GROUP BY sku),
c AS (SELECT *, SUM(rev) OVER (ORDER BY rev DESC ROWS UNBOUNDED PRECEDING) / SUM(rev) OVER () AS cum_share FROM r)
SELECT *, CASE WHEN cum_share <= 0.80 THEN 'A' WHEN cum_share <= 0.95 THEN 'B' ELSE 'C' END AS abc_class FROM c;

-- 4. Weekly SKU demand, zero-filled so weeks with no sales count as 0
CREATE OR REPLACE TABLE weekly AS
WITH weeks AS (SELECT DISTINCT week FROM sales),
skus AS (SELECT sku FROM abc),
grid AS (SELECT sku, week FROM skus CROSS JOIN weeks)
SELECT g.sku, g.week, COALESCE(SUM(s.qty),0) AS units, COALESCE(SUM(s.revenue),0) AS revenue
FROM grid g LEFT JOIN sales s USING (sku, week)
GROUP BY g.sku, g.week;
