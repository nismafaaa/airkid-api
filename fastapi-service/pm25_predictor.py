import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from datetime import timedelta
from typing import Optional


class PM25Predictor:
    def __init__(self, model_dir: str = None):
        base_dir = os.getcwd()
        self.model_dir = os.path.join(base_dir, "models") if model_dir is None else model_dir

        class CompatibleInputLayer(InputLayer):
            def __init__(self, batch_shape=None, input_shape=None, **kwargs):
                if batch_shape is not None and input_shape is None:
                    input_shape = batch_shape[1:]
                super().__init__(input_shape=input_shape, **kwargs)

        self.model = load_model(
            os.path.join(self.model_dir, "pm25_lstm_v1.h5"),
            custom_objects={
                "InputLayer": CompatibleInputLayer,
                "DTypePolicy": tf.keras.mixed_precision.Policy,
            },
            compile=False,
        )
        
        self.scaler = joblib.load(os.path.join(self.model_dir, "pm25_lstm_v1_scaler.joblib"))
        self.metadata = joblib.load(os.path.join(self.model_dir, "pm25_lstm_v1_metadata.joblib"))
        
        self.SEQ_LENGTH = self.metadata['sequence_length']
        self.FUTURE_DAYS = self.metadata['future_days']
        self.FEATURE_NAMES = self.metadata['feature_names']
        self.NUM_FEATURES = self.metadata['num_features']

        if (os.getenv("VERTEX_EXPORT_SAVEDMODEL", "false").lower() == "true"):
            try:
                self.export_saved_model()  
            except Exception:
                pass

    def _prepare_features(self, df):
        """Prepare features exactly as in your working version"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
   
        if 'pm25' not in df.columns:
            raise ValueError("DataFrame must contain a 'pm25' column.")
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour'] = getattr(df.index, 'hour', 0)
        
        for lag in [1, 2, 3, 7, 14]:
            df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
        
        for window in [3, 7, 14]:
            df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window).mean()
            df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window).std()
      
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
        
        last_seq = scaled[-self.SEQ_LENGTH:]
        X_input = np.expand_dims(last_seq, axis=0)
 
        y_pred_scaled = self.model.predict(X_input, verbose=0)

        temp = np.zeros((self.FUTURE_DAYS, self.NUM_FEATURES))
        temp[:, 0] = y_pred_scaled[0]
        y_pred_original = self.scaler.inverse_transform(temp)[:, 0]
        
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=self.FUTURE_DAYS, 
            freq='D'
        )
        
        return pd.DataFrame({"date": future_dates, "predicted_pm25": y_pred_original})
    
    def export_saved_model(self, export_dir: Optional[str] = None, overwrite: bool = True) -> str:
        """
        Export the loaded Keras .h5 model to TensorFlow SavedModel (saved_model.pb).
        Returns the export directory path.
        """
        # Decide default export dir:
        # 1) VERTEX_EXPORT_DIR if set
        # 2) models/tf_saved_model if models dir is writable
        # 3) /tmp/tf_saved_model as a safe writable fallback on Cloud Run
        if export_dir is None:
            env_dir = os.getenv("VERTEX_EXPORT_DIR")
            models_dir_candidate = os.path.join(self.model_dir, "tf_saved_model")
            parent_for_models = os.path.dirname(models_dir_candidate) or "."
            can_write_models = os.access(parent_for_models, os.W_OK)
            export_dir = env_dir or (models_dir_candidate if can_write_models else "/tmp/tf_saved_model")

        saved_pb = os.path.join(export_dir, "saved_model.pb")
        if os.path.isfile(saved_pb) and not overwrite:
            return export_dir

        os.makedirs(export_dir, exist_ok=True)
        try:
            self.model.save(export_dir, include_optimizer=False)
        except TypeError:
            tf.keras.models.save_model(self.model, export_dir, include_optimizer=False)
        return export_dir

    def get_saved_model_dir(self) -> Optional[str]:
        """
        Return a SavedModel directory if already exported (checks env, models/, /tmp), else None.
        """
        candidates = []
        env_dir = os.getenv("VERTEX_EXPORT_DIR")
        if env_dir:
            candidates.append(env_dir)
        candidates.append(os.path.join(self.model_dir, "tf_saved_model"))
        candidates.append("/tmp/tf_saved_model")
        for c in candidates:
            if c and os.path.isfile(os.path.join(c, "saved_model.pb")):
                return c
        return None