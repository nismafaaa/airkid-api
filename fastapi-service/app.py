from fastapi import FastAPI
import pandas as pd
import os
from pm25_predictor import PM25Predictor

app = FastAPI(title="PM2.5 Forecast API")

# Load predictor once at startup
predictor = PM25Predictor("models")

@app.get("/")
async def root():
    """Welcome message with API information"""
    return {
        "message": "PM2.5 Forecast API", 
        "usage": "Call GET /forecast to get predictions using sample data",
        "sequence_length": predictor.SEQ_LENGTH,
        "future_days": predictor.FUTURE_DAYS
    }

@app.get("/forecast")
async def forecast():
    """
    Get PM2.5 forecast using the sample data from models folder.
    No file upload required - automatically uses pm25_lstm_v1_sample_data.csv
    """
    try:
        # Load sample data from models folder
        sample_data_path = os.path.join("models", "pm25_lstm_v1_sample_data.csv")
        
        if not os.path.exists(sample_data_path):
            return {"error": f"Sample data file not found: {sample_data_path}"}
        
        # Read the sample data
        df = pd.read_csv(sample_data_path, parse_dates=['date'], index_col='date')
        
        # Get predictions
        preds = predictor.predict_next(df)
        
        # Format response
        results = [
            {
                "date": row["date"].strftime("%Y-%m-%d"), 
                "predicted_pm25": round(float(row["predicted_pm25"]), 2)
            }
            for _, row in preds.iterrows()
        ]
        
        return {
            "status": "success",
            "data_info": {
                "sample_data_shape": df.shape,
                "last_actual_date": df.index[-1].strftime("%Y-%m-%d"),
                "last_actual_pm25": round(float(df['pm25'].iloc[-1]), 2)
            },
            "forecast": results
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/data-info")
async def data_info():
    """Get information about the sample data"""
    try:
        sample_data_path = os.path.join("models", "pm25_lstm_v1_sample_data.csv")
        df = pd.read_csv(sample_data_path, parse_dates=['date'], index_col='date')
        
        return {
            "file_path": sample_data_path,
            "shape": df.shape,
            "date_range": {
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d")
            },
            "pm25_stats": {
                "mean": round(float(df['pm25'].mean()), 2),
                "min": round(float(df['pm25'].min()), 2),
                "max": round(float(df['pm25'].max()), 2),
                "latest": round(float(df['pm25'].iloc[-1]), 2)
            },
            "columns": list(df.columns)
        }
        
    except Exception as e:
        return {"error": f"Could not read sample data: {str(e)}"}