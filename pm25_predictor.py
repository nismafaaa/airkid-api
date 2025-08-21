import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from datetime import timedelta


class PM25Predictor:
    def __init__(self, model_dir: str = None):
        # Always resolve relative to current working directory
        base_dir = os.getcwd()
        self.model_dir = os.path.join(base_dir, "models") if model_dir is None else model_dir

        # Custom InputLayer to handle batch_shape compatibility
        class CompatibleInputLayer(InputLayer):
            def __init__(self, batch_shape=None, input_shape=None, **kwargs):
                if batch_shape is not None and input_shape is None:
                    input_shape = batch_shape[1:]
                super().__init__(input_shape=input_shape, **kwargs)

        # Load model with compatibility fixes; skip compiling to avoid optimizer deserialization
        self.model = load_model(
            os.path.join(self.model_dir, "pm25_lstm_v1.h5"),
            custom_objects={
                "InputLayer": CompatibleInputLayer,
                "DTypePolicy": tf.keras.mixed_precision.Policy,
            },
            compile=False,
        )
        
        # Load preprocessing objects & metadata (using correct filenames)
        self.scaler = joblib.load(os.path.join(self.model_dir, "pm25_lstm_v1_scaler.joblib"))
        self.metadata = joblib.load(os.path.join(self.model_dir, "pm25_lstm_v1_metadata.joblib"))
        
        # Extract metadata (using correct keys from your working version)
        self.SEQ_LENGTH = self.metadata['sequence_length']
        self.FUTURE_DAYS = self.metadata['future_days']
        self.FEATURE_NAMES = self.metadata['feature_names']
        self.NUM_FEATURES = self.metadata['num_features']

    def _prepare_features(self, df):
        """Prepare features exactly as in your working version"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        # Ensure pm25 column exists
        if 'pm25' not in df.columns:
            raise ValueError("DataFrame must contain a 'pm25' column.")
        
        # Calendar features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour'] = getattr(df.index, 'hour', 0)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
        
        # Rolling stats
        for window in [3, 7, 14]:
            df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window).mean()
            df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window).std()
        
        # Change features
        df['pm25_pct_change'] = df['pm25'].pct_change()
        df['pm25_diff'] = df['pm25'].diff()
        
        return df.dropna()

    def predict_next(self, df):
        """
        df: DataFrame with 'pm25' column and datetime index (at least SEQ_LENGTH days of data)
        Returns: DataFrame with future dates and predictions
        """
        df_processed = self._prepare_features(df)
        df_processed = df_processed[self.FEATURE_NAMES]
        scaled = self.scaler.transform(df_processed)
        
        if scaled.shape[0] < self.SEQ_LENGTH:
            raise ValueError(f"Need at least {self.SEQ_LENGTH} days of data for prediction")
        
        # Create sequence for LSTM
        last_seq = scaled[-self.SEQ_LENGTH:]
        X_input = np.expand_dims(last_seq, axis=0)
        
        # Predict
        y_pred_scaled = self.model.predict(X_input, verbose=0)
        
        # Inverse transform only pm25 column
        temp = np.zeros((self.FUTURE_DAYS, self.NUM_FEATURES))
        temp[:, 0] = y_pred_scaled[0]
        y_pred_original = self.scaler.inverse_transform(temp)[:, 0]
        
        # Return as DataFrame with future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=self.FUTURE_DAYS, 
            freq='D'
        )
        
        return pd.DataFrame({"date": future_dates, "predicted_pm25": y_pred_original})