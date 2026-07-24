# Beijing PM2.5 Forecasting (LSTM)

Predicts PM2.5 air pollution one hour ahead using an LSTM trained on the UCI Beijing PM2.5 dataset (43,824 hourly readings, 2010-2014). Inputs are the past 48 hours of PM2.5, dew point, temperature, pressure, wind direction, wind speed, and precipitation.

## Data and preprocessing

- Missing PM2.5 values (2,067 of them) were filled with linear interpolation rather than dropped or mean-filled, to preserve chronological order and avoid erasing seasonal patterns.
- Wind direction was label-encoded; all features were scaled to 0-1 with MinMaxScaler.
- Data was reshaped into sliding windows of 48 timesteps per sample for the LSTM input.
- Train/test split is sequential (80/20), not random, since shuffling would let the model see future data during training.

## Model

Single LSTM layer (50 units) -> Dropout (0.2) -> Dense (25, ReLU) -> Dense (1). Trained with Adam and MSE loss, early stopping on validation loss.

## Results

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² |
|---|---|---|---|
| Naive persistence (predict next hour = current hour) | 134.27 | 96.18 | -88.956 |
| LSTM | 2.22 | 1.80 | 0.975 |

The naive persistence baseline is included because hourly air quality changes slowly, so "no change" is a competitive prediction by default. The LSTM is only a meaningful result if it beats this baseline by a real margin, not just a small fraction of a percent.

## Limitations

- Forecasts only one hour ahead; multi-step forecasting would need a recursive or sequence-to-sequence setup.
- Hyperparameters (lookback window, units, layers) were chosen reasonably but not formally tuned.
- Single monitoring station and time period; would need retraining to generalize elsewhere.
- Results are from a single training run, not averaged across seeds.

## Stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn.
