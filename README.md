# Data-Driven Intelligent Reservoir Management for the Mahanadi Basin

Code for the paper *"Data-Driven Intelligent Reservoir Management for the Mahanadi Basin: LSTM-Based River-Flow Forecasting with a Ravishankar Sagar Operational Benchmark"*.

One-day-ahead river-discharge forecasting at the Mahanadi_Arang station, comparing Persistence, ARIMA(2,1,2), SVR, Random Forest and a two-layer LSTM (5 seeds) with RMSE, MAE, NSE, KGE and PBIAS. Forecast windows use only 15 consecutive calendar days, so no window crosses a data gap.

## Data
National Water Data Portal (NWIC), *River Discharge (Manual - Daily), Chhattisgarh Surface Water Department*.
Download the file `river_discharge_manual_daily_chhattisgarh_sw_cg_2001_2025.csv` from the portal and place it in the same folder as the scripts (the data file is not redistributed here).

## Run
```
pip install -r requirements.txt
python run.py     # trains and evaluates all models, writes results.json and test_predictions.csv
python figs.py    # writes fig1_discharge.png, fig2_obs_vs_lstm.png, fig3_rmse.png
```

## Files
- `common.py` – data loading and metrics (RMSE, MAE, NSE, KGE, PBIAS)
- `lstm_np.py` – two-layer LSTM implemented in NumPy (back-propagation through time, Adam, early stopping)
- `run.py` – experiment pipeline
- `figs.py` – figures
- `results.json`, `test_predictions.csv` – outputs reported in the paper

## Note
Python 3 with NumPy, SciPy, scikit-learn, pandas and Matplotlib. No new reservoir-optimization (NSGA-II) result is claimed; storage, release and demand data are not part of the dataset.
