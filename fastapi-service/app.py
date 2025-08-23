from fastapi import FastAPI
import pandas as pd
import os
from pm25_predictor import PM25Predictor
import asyncpg
from typing import Optional
from urllib.parse import urlparse, urlunparse
from datetime import datetime  # removed timezone

app = FastAPI(title="PM2.5 Forecast API")

predictor = PM25Predictor("models")

db_pool: Optional[asyncpg.pool.Pool] = None
last_db_error: Optional[str] = None
last_db_url_masked: Optional[str] = None

def _build_db_url_from_env() -> Optional[str]:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("PGHOST")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    port = os.getenv("PGPORT", "5432")
    if host and database and user:
        auth = f"{user}:{password}@" if password else f"{user}@"
        return f"postgresql://{auth}{host}:{port}/{database}"
    return None

def _get_db_connect_args():
    url = os.getenv("DATABASE_URL")
    if url:
        return {"dsn": url}
    host = os.getenv("PGHOST")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    port = int(os.getenv("PGPORT", "5432"))
    if host and database and user:
        args = {"host": host, "port": port, "user": user, "database": database}
        if password:
            args["password"] = password
        return args
    return None

def _mask_db_url(url: str) -> str:  
    try:
        p = urlparse(url)
        if p.username:
            netloc = f"{p.username}:***@{p.hostname}{(':' + str(p.port)) if p.port else ''}"
        else:
            netloc = p.netloc
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return "<masked>"

@app.on_event("startup")
async def on_startup():
    global db_pool, last_db_error, last_db_url_masked
    conn_args = _get_db_connect_args()
    if conn_args:
        if "dsn" in conn_args:
            last_db_url_masked = _mask_db_url(conn_args["dsn"])
        else:
            last_db_url_masked = f"postgresql://{conn_args.get('user')}:***@{conn_args.get('host')}:{conn_args.get('port')}/{conn_args.get('database')}"
        try:
            db_pool = await asyncpg.create_pool(min_size=1, max_size=5, **conn_args)
            last_db_error = None
        except Exception as e:
            last_db_error = str(e)
            db_pool = None

@app.on_event("shutdown")
async def on_shutdown():
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None

@app.get("/")
async def root():
    """Welcome message with API information"""
    return {
        "message": "PM2.5 Forecast API",
        "usage": "Use /get-forecast for provider/model results or /forecast-model to generate model forecasts",
        "db_usage": "GET /get-forecast (defaults: city='Malang, Indonesia', pollutant=pm25, 7 future days). Override with ?city=...&pollutant=...&horizon=...",
        "model_usage": "GET /forecast-model?persist=true",
        "db_configured": bool(db_pool),
        "sequence_length": predictor.SEQ_LENGTH,
        "future_days": predictor.FUTURE_DAYS
    }

@app.get("/get-forecast")
async def get_forecast(
    city: str = "Malang, Indonesia",
    pollutant: str = "pm25",
    horizon: Optional[int] = 7
):
    """
    Try source='model' first. If no rows, fall back to source='provider'.
    Includes the selected source in the response. Defaults to 7 future days for 'Malang, Indonesia'.
    """
    if not db_pool:
        return {"status": "unavailable", "message": "Database not configured. Use /forecast-model as fallback."}

    sql_base_no_source = """
    WITH latest AS (
      SELECT COALESCE(MAX(source_idx), -1) AS max_idx, MAX(created_at) AS max_created
      FROM aqi_forecast_daily
      WHERE city = $1 AND pollutant = $2
    ),
    rows AS (
      SELECT d.*, l.max_idx AS run_source_idx, l.max_created AS run_created_at
      FROM aqi_forecast_daily d
      CROSS JOIN latest l
      WHERE d.city = $1 AND d.pollutant = $2
        AND (
          (l.max_idx <> -1 AND d.source_idx = l.max_idx)
          OR (l.max_idx = -1 AND d.created_at >= l.max_created - INTERVAL '10 minutes')
        )
        AND d.day >= CURRENT_DATE
    )
    SELECT city, pollutant, day, avg, min, max, source_idx, created_at,
           run_source_idx, run_created_at
    FROM rows
    ORDER BY day ASC
    LIMIT COALESCE($3::int, 7);
    """

    sql_base_with_source = """
    WITH latest AS (
      SELECT COALESCE(MAX(source_idx), -1) AS max_idx, MAX(created_at) AS max_created
      FROM aqi_forecast_daily
      WHERE city = $1 AND pollutant = $2
    ),
    rows AS (
      SELECT d.*, l.max_idx AS run_source_idx, l.max_created AS run_created_at
      FROM aqi_forecast_daily d
      CROSS JOIN latest l
      WHERE d.city = $1 AND d.pollutant = $2
        AND (
          (l.max_idx <> -1 AND d.source_idx = l.max_idx)
          OR (l.max_idx = -1 AND d.created_at >= l.max_created - INTERVAL '10 minutes')
        )
        AND d.day >= CURRENT_DATE
    )
    SELECT city, pollutant, source, day, avg, min, max, source_idx, created_at,
           run_source_idx, run_created_at
    FROM rows
    ORDER BY day ASC
    LIMIT COALESCE($3::int, 7);
    """

    sql_with_source = """
    WITH latest AS (
      SELECT COALESCE(MAX(source_idx), -1) AS max_idx, MAX(created_at) AS max_created
      FROM aqi_forecast_daily
      WHERE city = $1 AND pollutant = $2 AND source = $3
    ),
    rows AS (
      SELECT d.*, l.max_idx AS run_source_idx, l.max_created AS run_created_at
      FROM aqi_forecast_daily d
      CROSS JOIN latest l
      WHERE d.city = $1 AND d.pollutant = $2 AND d.source = $3
        AND (
          (l.max_idx <> -1 AND d.source_idx = l.max_idx)
          OR (l.max_idx = -1 AND d.created_at >= l.max_created - INTERVAL '10 minutes')
        )
        AND d.day >= CURRENT_DATE
    )
    SELECT city, pollutant, source, day, avg, min, max, source_idx, created_at,
           run_source_idx, run_created_at
    FROM rows
    ORDER BY day ASC
    LIMIT COALESCE($4::int, 7);
    """

    try:
        async with db_pool.acquire() as conn:
            has_source_col = await conn.fetchval("""
              SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'aqi_forecast_daily'
                  AND column_name = 'source'
              );
            """)

            rows = []
            used_source: Optional[str] = None

            if has_source_col:
                for s in ("model", "provider"):
                    rows = await conn.fetch(sql_with_source, city, pollutant, s, horizon)
                    if rows:
                        used_source = s
                        break
            else:
                rows = await conn.fetch(sql_base_no_source, city, pollutant, horizon)

        if not rows:
            return {"status": "empty", "city": city, "pollutant": pollutant, "forecast": []}

        run_source_idx = rows[0]["run_source_idx"]
        run_created_at = rows[0]["run_created_at"].isoformat() if rows[0]["run_created_at"] else None

        forecast = [
            {"day": r["day"].isoformat(), "avg": r["avg"], "min": r["min"], "max": r["max"]}
            for r in rows
        ]

        run_info = {"source_idx": run_source_idx, "created_at": run_created_at}
        run_info["source"] = used_source if has_source_col else "unknown"

        return {
            "status": "success",
            "city": city,
            "pollutant": pollutant,
            "run": run_info,
            "horizon": len(forecast),
            "forecast": forecast
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/db-health")
async def db_health():
    env_info = {
        "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
        "PGHOST": os.getenv("PGHOST"),
        "PGPORT": os.getenv("PGPORT") or "5432",
        "PGDATABASE": os.getenv("PGDATABASE"),
        "PGUSER": os.getenv("PGUSER"),
        "PGPASSWORD": bool(os.getenv("PGPASSWORD")),
    }
    computed_args = _get_db_connect_args()
    if computed_args and "dsn" in computed_args:
        masked = _mask_db_url(computed_args["dsn"])
        connect_mode = "dsn"
    elif computed_args:
        masked = f"postgresql://{computed_args.get('user')}:***@{computed_args.get('host')}:{computed_args.get('port')}/{computed_args.get('database')}"
        connect_mode = "params"
    else:
        masked = None
        connect_mode = None

    info = {
        "configured": bool(db_pool),
        "env": env_info,
        "db_url_present": bool(_build_db_url_from_env()),
        "connect_mode": connect_mode,
        "db_url_masked": masked,
        "startup_db_url_masked": last_db_url_masked,
        "last_error": last_db_error,
    }
    if not db_pool:
        info["message"] = "No DATABASE_URL/PG* env at startup or connection failed."
        return info
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        info["can_connect"] = True
    except Exception as e:
        info["can_connect"] = False
        info["error"] = str(e)
    return info

async def _fetch_daily_pm25_history(conn: asyncpg.Connection, city: str):
    sql = """
      SELECT (observed_at AT TIME ZONE 'UTC')::date AS day, AVG(pm25) AS pm25
      FROM aqi_observations
      WHERE city = $1 AND pm25 IS NOT NULL
      GROUP BY day
      ORDER BY day
    """
    return await conn.fetch(sql, city)

def _build_daily_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame indexed by date with column 'pm25' (daily)
    Output: DataFrame with columns matching the CSV feature set.
    """
    df = daily_df.sort_index().copy()
    features = pd.DataFrame(index=df.index)

    features["pm25"] = df["pm25"]

    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["quarter"] = features.index.quarter
    features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
    features["hour"] = 0  

    features["pm25_lag_1"] = df["pm25"].shift(1)
    features["pm25_lag_2"] = df["pm25"].shift(2)
    features["pm25_lag_3"] = df["pm25"].shift(3)
    features["pm25_lag_7"] = df["pm25"].shift(7)
    features["pm25_lag_14"] = df["pm25"].shift(14)

    features["pm25_rolling_mean_3"] = df["pm25"].rolling(3, min_periods=3).mean()
    features["pm25_rolling_std_3"] = df["pm25"].rolling(3, min_periods=3).std()
    features["pm25_rolling_mean_7"] = df["pm25"].rolling(7, min_periods=7).mean()
    features["pm25_rolling_std_7"] = df["pm25"].rolling(7, min_periods=7).std()
    features["pm25_rolling_mean_14"] = df["pm25"].rolling(14, min_periods=14).mean()
    features["pm25_rolling_std_14"] = df["pm25"].rolling(14, min_periods=14).std()

    features["pm25_pct_change"] = df["pm25"].pct_change()
    features["pm25_diff"] = df["pm25"].diff()

    features = features.dropna()

    cols = [
        "pm25",
        "day_of_week", "month", "quarter", "is_weekend", "hour",
        "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_7", "pm25_lag_14",
        "pm25_rolling_mean_3", "pm25_rolling_std_3",
        "pm25_rolling_mean_7", "pm25_rolling_std_7",
        "pm25_rolling_mean_14", "pm25_rolling_std_14",
        "pm25_pct_change", "pm25_diff",
    ]
    return features[cols]

@app.get("/forecast-model")
async def forecast_model(city: str = "Malang, Indonesia", persist: bool = False):
    """
    Use latest aqi_observations (daily avg) as model input.
    Returns 7-day forecast with source='model'. If persist=true, upserts into aqi_forecast_daily.
    """
    if not db_pool:
        return {"status": "unavailable", "message": "Database not configured."}

    try:
        async with db_pool.acquire() as conn:
            rows = await _fetch_daily_pm25_history(conn, city)
            if not rows:
                return {"status": "empty", "message": "No observations found.", "city": city}

        df = pd.DataFrame([{"date": r["day"], "pm25": float(r["pm25"])} for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_idx)
        df["pm25"] = df["pm25"].interpolate(method="time").ffill().bfill()

        feats = _build_daily_features(df)

        required = predictor.SEQ_LENGTH
        available_feats = int(len(feats))
        available_raw = int(len(df))

        if max(available_feats, available_raw) < required:
            return {
                "status": "insufficient_history",
                "required_days": required,
                "available_days_features": available_feats,
                "available_days_raw": available_raw,
                "city": city
            }

        preds = None
        input_mode = None
        first_err = None

        try:
            input_df = feats
            preds = predictor.predict_next(input_df)
            input_mode = "features"
        except Exception as e:
            first_err = str(e)
            try:
                input_df = df[["pm25"]]
                preds = predictor.predict_next(input_df)
                input_mode = "raw_pm25"
            except Exception as e2:
                return {
                    "status": "error",
                    "message": first_err,
                    "fallback_error": str(e2),
                    "available_days_features": available_feats,
                    "available_days_raw": available_raw
                }

        if "date" in preds.columns:
            preds = preds.head(7)
            dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in preds["date"]]
            values = preds["predicted_pm25"].tolist()
        else:
            horizon = min(7, len(preds))
            start_date = df.index.max() + pd.Timedelta(days=1)
            dates = pd.date_range(start=start_date, periods=horizon, freq="D").strftime("%Y-%m-%d").tolist()
            if hasattr(preds, "columns") and "predicted_pm25" in preds.columns:
                values = preds["predicted_pm25"].tolist()
            else:
                values = list(preds[:horizon])

        forecast = [
            {"day": d, "avg": int(round(float(v))), "min": None, "max": None}
            for d, v in zip(dates, values)
        ]

        required = predictor.SEQ_LENGTH
        if input_mode == "features" and len(feats) >= required:
            win_start, win_end = feats.index[-required], feats.index[-1]
        else:
            win_start, win_end = df.index[-required], df.index[-1]

        persisted = 0
        if persist and forecast:
            upsert_sql = """
              INSERT INTO aqi_forecast_daily (city, pollutant, day, avg, min, max, source, source_idx)
              VALUES ($1, 'pm25', $2, $3, $4, $5, 'model', NULL)
              ON CONFLICT (city, pollutant, day, source)
              DO UPDATE SET avg = EXCLUDED.avg,
                            min = EXCLUDED.min,
                            max = EXCLUDED.max,
                            created_at = NOW()
            """
            async with db_pool.acquire() as conn:
                await conn.executemany(
                    upsert_sql,
                    [
                        (
                            city,
                            datetime.strptime(item["day"], "%Y-%m-%d").date(),
                            item["avg"],
                            item["min"],
                            item["max"],
                        )
                        for item in forecast
                    ],
                )
                persisted = len(forecast)

        return {
            "status": "success",
            "city": city,
            "source": "model",
            "model_input": input_mode,
            "horizon": len(forecast),
            "available_days_features": available_feats,
            "available_days_raw": available_raw,
            "used_history": {
                "start": pd.to_datetime(win_start).strftime("%Y-%m-%d"),
                "end": pd.to_datetime(win_end).strftime("%Y-%m-%d"),
                "days": int(required),
            },
            "persisted": persisted if persist else 0,
            "forecast": forecast,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}