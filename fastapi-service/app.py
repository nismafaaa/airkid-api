from fastapi import FastAPI, HTTPException
import pandas as pd
import os
from pm25_predictor import PM25Predictor
import asyncpg
from typing import Optional
from urllib.parse import urlparse, urlunparse
from datetime import datetime, timedelta
import logging
from google.cloud import aiplatform
from google.cloud import logging as gcloud_logging
from google.cloud import monitoring_v3
from google.api import metric_pb2, monitored_resource_pb2
from google.protobuf import timestamp_pb2
from pydantic import BaseModel
from typing import List, Union, Dict, Any, Optional
import json
from dotenv import load_dotenv
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import httpx

try:
    import google.generativeai as genai
except Exception:
    genai = None

app = FastAPI(title="PM2.5 Forecast API")

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://airkid.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PM25Predictor("models")

db_pool: Optional[asyncpg.pool.Pool] = None
last_db_error: Optional[str] = None
last_db_url_masked: Optional[str] = None

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION")
GCS_BUCKET = os.getenv("GCS_BUCKET")  
VERTEX_REGISTER_ON_STARTUP = os.getenv("VERTEX_REGISTER_ON_STARTUP", "false").lower() == "true"
VERTEX_PREDICTION_IMAGE = os.getenv("VERTEX_PREDICTION_IMAGE") or "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
MODEL_NAME = os.getenv("MODEL_NAME", "pm25-predictor")
MODEL_VERSION = os.getenv("MODEL_VERSION")  
_vertex_model_resource: Optional[str] = None
_monitoring_client: Optional[monitoring_v3.MetricServiceClient] = None
last_vertex_error: Optional[str] = None  
VERTEX_SKIP_REGISTER_IF_EXISTS = os.getenv("VERTEX_SKIP_REGISTER_IF_EXISTS", "true").lower() == "true"  # new

GEMINI_API_KEY = _read_env_or_file("GEMINI_API_KEY")
EXTERNAL_LATEST_OBS_URL = os.getenv(
    "EXTERNAL_LATEST_OBS_URL",
    "https://fastapi-service-641497729175.asia-southeast2.run.app/latest-observation?city=Malang%2C%20Indonesia",
)
# Log non-secret config at import/init time
logging.info({
    "event": "config_init",
    "gemini_configured": bool(GEMINI_API_KEY),
    "external_obs_url": EXTERNAL_LATEST_OBS_URL,
})

SUPPORTED_VERTEX_FILES = {
    "model.pkl", "model.joblib", "model.bst", "model.mar", "saved_model.pb", "saved_model.pbtxt"
}

def _read_env_or_file(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Returns secret from ENV[name] or, if empty, from ENV[name+'_FILE'] path.
    """
    val = os.getenv(name, default)
    if not val:
        file_path = os.getenv(f"{name}_FILE")
        if file_path and os.path.isfile(file_path):
            try:
                val = Path(file_path).read_text(encoding="utf-8").strip()
            except Exception:
                val = default
    return val

def _find_vertex_artifact_path(base_dir: str) -> Optional[str]:
    """
    Recursively search for a supported artifact. For SavedModel, return its folder.
    For file-based models, return the file path.
    """
    try:
        for root, _, files in os.walk(base_dir):
            if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                return root
            for f in files:
                if f in ("model.pkl", "model.joblib", "model.bst", "model.mar"):
                    return os.path.join(root, f)
    except Exception:
        return None
    return None

def _get_vertex_runtime_staging_bucket() -> Optional[str]:
    """Return the SDK's currently configured staging bucket."""
    try:
        return getattr(aiplatform.initializer.global_config, "staging_bucket", None)
    except Exception:
        return None

def _display_name(display_name_override: Optional[str]) -> str:
    """Consistent display_name builder."""
    return display_name_override or f"{MODEL_NAME}{('-' + MODEL_VERSION) if MODEL_VERSION else ''}"

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

def _init_gcp():
    global _monitoring_client
    if not GCP_PROJECT:
        return
    try:
        gcloud_logging.Client(project=GCP_PROJECT).setup_logging()  
    except Exception:
        pass
    try:
        staging = None
        if GCS_BUCKET:
            staging = GCS_BUCKET if GCS_BUCKET.startswith("gs://") else f"gs://{GCS_BUCKET}"
        aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION, staging_bucket=staging)
        logging.info({
            "event": "vertex_init",
            "project": GCP_PROJECT,
            "location": GCP_LOCATION,
            "staging_bucket_env": staging,
            "staging_bucket_runtime": _get_vertex_runtime_staging_bucket(),
        })
    except Exception as e:
        logging.warning({"event": "vertex_init_failed", "error": str(e)})
    try:
        _monitoring_client = monitoring_v3.MetricServiceClient()
    except Exception:
        _monitoring_client = None

def _register_model_in_vertex(models_dir: str = "models", display_name: Optional[str] = None, artifact_path: Optional[str] = None):
    global _vertex_model_resource, last_vertex_error
    if not GCP_PROJECT:
        last_vertex_error = "GCP_PROJECT not set"
        return {"status": "error", "error": last_vertex_error}

    try:
        dn = _display_name(display_name)

        if VERTEX_SKIP_REGISTER_IF_EXISTS:
            try:
                existing = aiplatform.Model.list(filter=f'display_name="{dn}"')
                if existing:
                    _vertex_model_resource = existing[0].resource_name
                    last_vertex_error = None
                    logging.info({"event": "vertex_model_exists", "model": _vertex_model_resource, "display_name": dn})
                    return {"status": "exists", "model_resource": _vertex_model_resource, "display_name": dn}
            except Exception as list_err:
                logging.warning({"event": "vertex_list_failed", "error": str(list_err)})

        local_dir = models_dir
        if not (local_dir and os.path.isdir(local_dir)):
            last_vertex_error = f"models_dir not found: {local_dir}"
            logging.warning({"event": "vertex_model_register_failed", "error": last_vertex_error})
            return {"status": "error", "error": last_vertex_error}

        has_files = any(True for _ in os.scandir(local_dir))
        if not has_files:
            last_vertex_error = f"models_dir is empty: {local_dir}"
            logging.warning({"event": "vertex_model_register_failed", "error": last_vertex_error})
            return {"status": "error", "error": last_vertex_error}

        artifact_uri = artifact_path or _find_vertex_artifact_path(local_dir) or _find_vertex_artifact_path("/tmp")
        if not artifact_uri:
            try:
                export_dir = predictor.get_saved_model_dir()
                if not export_dir:
                    export_dir = os.getenv("VERTEX_EXPORT_DIR") or "/tmp/tf_saved_model"
                    export_dir = predictor.export_saved_model(export_dir=export_dir, overwrite=True)
                if os.path.isfile(os.path.join(export_dir, "saved_model.pb")):
                    artifact_uri = export_dir
            except Exception as export_err:
                logging.warning({"event": "vertex_export_failed", "error": str(export_err)})

        if not artifact_uri:
            last_vertex_error = (
                f"artifact_uri directory does not contain supported files: {sorted(list(SUPPORTED_VERTEX_FILES))}. "
                f"Put one of these under {local_dir} or /tmp (recursively), "
                f"set VERTEX_EXPORT_DIR to a writable path (e.g. /tmp/tf_saved_model), "
                f"or pass ?artifact_path=/path/to/file_or_dir"
            )
            logging.warning({"event": "vertex_model_register_failed", "error": last_vertex_error})
            return {"status": "error", "error": last_vertex_error}

        model = aiplatform.Model.upload(
            display_name=dn,
            artifact_uri=artifact_uri,  
            serving_container_image_uri=VERTEX_PREDICTION_IMAGE,
            description="PM2.5 predictor registered from FastAPI service",
            labels={"app": "pm25-forecast"},
        )
        _vertex_model_resource = model.resource_name
        last_vertex_error = None
        logging.info({"event": "vertex_model_registered", "model": _vertex_model_resource, "display_name": dn, "artifact_uri": artifact_uri})
        return {"status": "success", "model_resource": _vertex_model_resource, "display_name": dn, "artifact_uri": artifact_uri}
    except Exception as e:
        last_vertex_error = str(e)
        logging.warning({"event": "vertex_model_register_failed", "error": last_vertex_error})
        return {"status": "error", "error": last_vertex_error}

def _log_forecast_event(city: str, pollutant: str, input_mode: str, forecast: list, persisted: int):
    try:
        logging.info({
            "event": "forecast_model_run",
            "city": city,
            "pollutant": pollutant,
            "horizon": len(forecast),
            "first_day": forecast[0]["day"] if forecast else None,
            "last_day": forecast[-1]["day"] if forecast else None,
            "avg_mean": float(pd.Series([f["avg"] for f in forecast]).mean()) if forecast else None,
            "persisted": persisted,
            "model_input": input_mode,
            "vertex_model": _vertex_model_resource,
        })
    except Exception:
        pass

def _ensure_metric_descriptor(metric_type: str, value_type: metric_pb2.MetricDescriptor.ValueType, unit: str, description: str):
    if not _monitoring_client or not GCP_PROJECT:
        return
    name = f"projects/{GCP_PROJECT}"
    md = metric_pb2.MetricDescriptor()
    md.type = metric_type
    md.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    md.value_type = value_type
    md.unit = unit
    md.description = description
    md.display_name = metric_type.split("/")[-1]
    try:
        _monitoring_client.create_metric_descriptor(name=name, metric_descriptor=md)
    except Exception:
        pass

def _publish_forecast_metrics(city: str, forecast: list):
    if not _monitoring_client or not GCP_PROJECT:
        return
    _ensure_metric_descriptor("custom.googleapis.com/pm25/forecast/horizon", metric_pb2.MetricDescriptor.ValueType.INT64, "1", "PM2.5 forecast horizon length")
    _ensure_metric_descriptor("custom.googleapis.com/pm25/forecast/avg_mean", metric_pb2.MetricDescriptor.ValueType.DOUBLE, "1", "Mean of forecasted PM2.5 avg values")
    now = datetime.utcnow()
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(now)
    interval = monitoring_v3.TimeInterval(end_time=ts)
    resource = monitored_resource_pb2.MonitoredResource(type="global", labels={"project_id": GCP_PROJECT})
    horizon = len(forecast)
    series1 = monitoring_v3.TimeSeries()
    series1.metric.type = "custom.googleapis.com/pm25/forecast/horizon"
    series1.metric.labels["city"] = city
    series1.resource.CopyFrom(resource)
    point1 = monitoring_v3.Point()
    point1.interval.CopyFrom(interval)
    point1.value.int64_value = int(horizon)
    series1.points.append(point1)
    mean_val = float(pd.Series([f["avg"] for f in forecast]).mean()) if forecast else 0.0
    series2 = monitoring_v3.TimeSeries()
    series2.metric.type = "custom.googleapis.com/pm25/forecast/avg_mean"
    series2.metric.labels["city"] = city
    series2.resource.CopyFrom(resource)
    point2 = monitoring_v3.Point()
    point2.interval.CopyFrom(interval)
    point2.value.double_value = mean_val
    series2.points.append(point2)
    try:
        _monitoring_client.create_time_series(name=f"projects/{GCP_PROJECT}", time_series=[series1, series2])
    except Exception:
        pass

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
    try:
        _init_gcp()
        if VERTEX_REGISTER_ON_STARTUP:
            _register_model_in_vertex("models")
    except Exception:
        pass

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
        "future_days": predictor.FUTURE_DAYS,
        "vertex": {
            "project": GCP_PROJECT,
            "location": GCP_LOCATION,
            "staging_bucket": (GCS_BUCKET if GCS_BUCKET and GCS_BUCKET.startswith("gs://") else (f"gs://{GCS_BUCKET}" if GCS_BUCKET else None)),
            "model_resource": _vertex_model_resource
        }
    }

@app.get("/get-forecast",tags=["external"])
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
        AND d.day > CURRENT_DATE
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
        AND d.day > CURRENT_DATE
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
        AND d.day > CURRENT_DATE
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

    today_date = datetime.utcnow().date()

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

        desired_horizon = 7
        series_next = df.index.max().date() + timedelta(days=1)
        start_date = max(series_next, today_date + timedelta(days=1))

        if "date" in preds.columns:
            preds = preds.sort_values(by="date").copy()
            preds["date"] = pd.to_datetime(preds["date"]).dt.date
            filt = preds[preds["date"] >= start_date]

            dates = [d.strftime("%Y-%m-%d") for d in filt["date"].iloc[:desired_horizon]]
            values = filt["predicted_pm25"].iloc[:desired_horizon].tolist()

            if len(dates) < desired_horizon:
                if not dates:
                    last_date = start_date - timedelta(days=1)
                    last_val = preds["predicted_pm25"].iloc[-1]
                else:
                    last_date = datetime.strptime(dates[-1], "%Y-%m-%d").date()
                    last_val = values[-1]
                while len(dates) < desired_horizon:
                    last_date = last_date + timedelta(days=1)
                    dates.append(last_date.strftime("%Y-%m-%d"))
                    values.append(last_val)
        else:

            k = (start_date - series_next).days  
            if hasattr(preds, "columns") and "predicted_pm25" in preds.columns:
                seq = preds["predicted_pm25"].tolist()
            else:
                seq = list(preds)

            seq = seq[k:] if k > 0 else seq

            if len(seq) < desired_horizon and len(seq) > 0:
                last_val = seq[-1]
                seq = seq + [last_val] * (desired_horizon - len(seq))

            values = seq[:desired_horizon]
            dates = pd.date_range(
                start=pd.Timestamp(start_date),
                periods=desired_horizon,
                freq="D"
            ).strftime("%Y-%m-%d").tolist()

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
              VALUES ($1, 'pm25', $2, $3, $4, $5, $6, $7, NULL)
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
                            "model"
                        )
                        for item in forecast
                    ],
                )
                persisted = len(forecast)

        try:
            _log_forecast_event(city=city, pollutant="pm25", input_mode=input_mode, forecast=forecast, persisted=persisted if persist else 0)
            _publish_forecast_metrics(city=city, forecast=forecast)
        except Exception:
            pass

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

@app.get("/vertex-info")
async def vertex_info():
    candidate_models = _find_vertex_artifact_path("models") if os.path.isdir("models") else None
    candidate_tmp = _find_vertex_artifact_path("/tmp")
    saved_model_dir = None
    try:
        saved_model_dir = predictor.get_saved_model_dir()
    except Exception:
        pass
    return {
        "project": GCP_PROJECT,
        "location": GCP_LOCATION,
        "staging_bucket": (GCS_BUCKET if GCS_BUCKET and GCS_BUCKET.startswith("gs://") else (f"gs://{GCS_BUCKET}" if GCS_BUCKET else None)),
        "runtime_staging_bucket": _get_vertex_runtime_staging_bucket(),
        "prediction_image": VERTEX_PREDICTION_IMAGE,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "register_on_startup": VERTEX_REGISTER_ON_STARTUP,
        "skip_register_if_exists": VERTEX_SKIP_REGISTER_IF_EXISTS,
        "model_resource": _vertex_model_resource,
        "last_error": last_vertex_error,
        "candidate_artifact_models": candidate_models,
        "candidate_artifact_tmp": candidate_tmp,
        "saved_model_dir": saved_model_dir,
        "supported_files": sorted(list(SUPPORTED_VERTEX_FILES)),
        "hints": [
            "If runtime_staging_bucket != staging_bucket, call /vertex/reinit?bucket=gs://<new-bucket>&location=<region>.",
            "Update Cloud Run env GCS_BUCKET to persist across restarts."
        ],
    }

@app.post("/vertex-reinit")
async def vertex_reinit(bucket: Optional[str] = None, location: Optional[str] = None):
    """
    Reinitialize Vertex AI SDK with a new staging bucket and/or location at runtime.
    Example: POST /vertex/reinit?bucket=gs://airkid-aqi-vertex-ase2&location=asia-southeast2
    """
    global GCS_BUCKET, GCP_LOCATION
    new_bucket = bucket or GCS_BUCKET
    if new_bucket and not new_bucket.startswith("gs://"):
        new_bucket = f"gs://{new_bucket}"
    new_location = location or GCP_LOCATION
    try:
        aiplatform.init(project=GCP_PROJECT, location=new_location, staging_bucket=new_bucket)
        if bucket:
            GCS_BUCKET = new_bucket
        if location:
            GCP_LOCATION = new_location
        return {
            "status": "ok",
            "project": GCP_PROJECT,
            "location": GCP_LOCATION,
            "staging_bucket": GCS_BUCKET,
            "runtime_staging_bucket": _get_vertex_runtime_staging_bucket(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/vertex-register")
async def vertex_register(models_dir: str = "models", display_name: Optional[str] = None, artifact_path: Optional[str] = None):
    """
    Trigger Vertex Model upload. Optionally override artifact_path (file or SavedModel directory).
    """
    return _register_model_in_vertex(models_dir=models_dir, display_name=display_name, artifact_path=artifact_path)

@app.get("/latest-observation", tags=["external"])
async def latest_observation(city: str = "Malang, Indonesia"):
    """
    Return the latest available values for pm25, temp, wind, and humidity.
    Tries city-specific first, then falls back to global latest non-null.
    """
    if not db_pool:
        return {"status": "unavailable", "message": "Database not configured."}

    metric_to_column = {
        "pm25": "pm25",
        "temp": "temperature",
        "wind": "wind",
        "humidity": "humidity",
    }
    metrics = ["pm25", "temp", "wind", "humidity"]

    async def _fetch_latest_metric_value(conn: asyncpg.Connection, col: str, city_name: str):
        row = await conn.fetchrow(
            f"""
            SELECT {col} AS value, observed_at
            FROM aqi_observations
            WHERE city = $1 AND {col} IS NOT NULL
            ORDER BY observed_at DESC
            LIMIT 1
            """,
            city_name,
        )
        if row:
            return row["value"], row["observed_at"]
        row = await conn.fetchrow(
            f"""
            SELECT {col} AS value, observed_at
            FROM aqi_observations
            WHERE {col} IS NOT NULL
            ORDER BY observed_at DESC
            LIMIT 1
            """
        )
        if row:
            return row["value"], row["observed_at"]
        return None, None

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'aqi_observations'
                  AND column_name = ANY($1::text[])
                """,
                list(set(metric_to_column.values())),
            )
            existing_cols = {r["column_name"] for r in rows}

            data = {}
            latest_ts = None

            for key in metrics:
                col = metric_to_column.get(key, key)
                if col not in existing_cols:
                    data[key] = {"value": None, "observed_at": None}
                    continue

                raw_val, ts = await _fetch_latest_metric_value(conn, col, city)

                if raw_val is None:
                    norm_val = None
                else:
                    if key == "pm25":
                        norm_val = int(round(float(raw_val)))
                    else:
                        norm_val = float(raw_val)

                data[key] = {
                    "value": norm_val,
                    "observed_at": ts.isoformat() if ts else None,
                }
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts

            status = "success" if any(v["value"] is not None for v in data.values()) else "empty"
            return {
                "status": status,
                "city": city,
                "data": data,
                "latest_observed_at": latest_ts.isoformat() if latest_ts else None,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def _fetch_latest_observation_external() -> Dict[str, Any]:
    """
    Fetch latest observation from the external endpoint (Malang fixed).
    """
    try:
        logging.info({"event": "external_obs_request", "url": EXTERNAL_LATEST_OBS_URL})
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(EXTERNAL_LATEST_OBS_URL)
            resp.raise_for_status()
            data = resp.json()
            logging.info({
                "event": "external_obs_response",
                "status_code": resp.status_code,
                "latest_observed_at": data.get("latest_observed_at"),
                "pm25": data.get("data", {}).get("pm25", {}).get("value"),
            })
            return data
    except httpx.HTTPError as e:
        logging.warning({"event": "external_obs_http_error", "error": str(e)})
        raise HTTPException(status_code=502, detail=f"Failed to fetch external latest observation: {str(e)}")
    except Exception as e:
        logging.exception("external_obs_unknown_error")
        raise HTTPException(status_code=502, detail=f"Failed to fetch external latest observation: {str(e)}")

# -------------------- Activity recommendation models and helpers --------------------
class UserProfile(BaseModel):
    childName: str
    childAge: Union[str, int]
    healthSensitivities: List[str] = []
    activityPreferences: List[str] = []

class ActivityRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    city: Optional[str] = "Malang, Indonesia"

def _pm25_to_aqi(conc: float) -> Dict[str, Any]:
    """
    Convert PM2.5 (µg/m3) to AQI using US EPA breakpoints.
    Returns: { aqi: int, level: str }
    """
    try:
        c = max(0.0, float(conc))
    except Exception:
        return {"aqi": None, "level": None}
    bps = [
        (0.0, 12.0, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate Caution"),
        (35.5, 55.4, 101, 150, "Caution for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy Caution"),
        (150.5, 250.4, 201, 300, "Very Unhealthy - Avoid Outdoor"),
        (250.5, 350.4, 301, 400, "Hazardous - Stay Indoors"),
        (350.5, 500.4, 401, 500, "Hazardous - Stay Indoors"),
    ]
    for Cl, Ch, Il, Ih, lvl in bps:
        if c <= Ch:
            aqi = int(round((Ih - Il) / (Ch - Cl) * (c - Cl) + Il))
            return {"aqi": aqi, "level": lvl}
    return {"aqi": 500, "level": "Hazardous - Stay Indoors"}

def _build_activity_prompt(req: ActivityRequest, obs: Dict[str, Any], aqi_info: Dict[str, Any]) -> str:
    city = "Malang, Indonesia"
    pm25_val = obs.get("data", {}).get("pm25", {}).get("value")
    temp_val = obs.get("data", {}).get("temp", {}).get("value")
    wind_val = obs.get("data", {}).get("wind", {}).get("value")
    hum_val = obs.get("data", {}).get("humidity", {}).get("value")
    latest_ts = obs.get("latest_observed_at")

    sensitivities = ", ".join(req.user_profile.healthSensitivities) or "None specified"
    prefs = ", ".join(req.user_profile.activityPreferences) or "No specific preferences"

    guidance = f"""
Anda adalah asisten yang merekomendasikan aktivitas ramah anak dengan mempertimbangkan kualitas udara saat ini.
Balas HANYA berupa objek JSON ketat (tanpa markdown, tanpa komentar). Gunakan skema JSON berikut:
{{
  "recommendation_level": "string",
  "summary": "string",
  "recommended_activity": {{
    "name": "string",
    "location_name": "string",
    "developmental_benefit": "string",
    "safety_tip": "string"
  }},
  "current_aqi": number
}}

Konteks:
- Kota: {city}
- Anak: name="{req.user_profile.childName}", age="{req.user_profile.childAge}"
- Sensitivitas kesehatan: {sensitivities}
- Preferensi: {prefs}
- Waktu observasi terbaru: {latest_ts}
- Metrik terbaru: pm25={pm25_val}, temp={temp_val}, wind={wind_val}, humidity={hum_val}
- AQI terhitung (dari PM2.5): {aqi_info.get("aqi")} ({aqi_info.get("level")})

Instruksi:
- Tulis SELURUH jawaban dalam Bahasa Indonesia.
- Rekomendasikan tempat/aktivitas YANG HANYA berada di Kota Malang (contoh: Alun-Alun Malang, taman kota, museum/ruang bermain dalam ruangan di Malang). Jangan rekomendasikan lokasi di luar Malang.
- Sesuaikan dengan usia dan sensitivitas (contoh: asma -> durasi luar ruang lebih singkat atau opsi indoor jika AQI tinggi).
- Buat "summary" singkat dan hangat.
- Selaraskan "recommendation_level" dengan tingkat AQI di atas.
- Jika AQI bukan "Good", batasi durasi luar ruang dan berikan tips keselamatan yang praktis.
- Pastikan "recommended_activity.location_name" menyebut lokasi di Malang.
- Balas dengan JSON valid dan ringkas (minified).
""".strip()
    return guidance

def _parse_gemini_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        lines = [l for l in txt.splitlines() if not l.strip().startswith("```")]
        txt = "\n".join(lines).strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start >= 0 and end > start:
                return json.loads(txt[start:end+1])
        except Exception:
            return None

@app.post("/recommend-activity", tags=["external"])
async def recommend_activity(req: ActivityRequest):
    """
    Recommend an activity using Gemini, enriched with latest-observation.
    If user_profile is missing, fill it with defaults. No rule-based fallback.
    """
    obs = await _fetch_latest_observation_external()

    pm25_val = None
    try:
        pm25_val = obs.get("data", {}).get("pm25", {}).get("value")
    except Exception:
        pm25_val = None
    aqi_info = _pm25_to_aqi(pm25_val) if pm25_val is not None else {"aqi": None, "level": None}

    default_profile = UserProfile(
        childName="Budi",
        childAge="6",
        healthSensitivities=["Asthma or other respiratory sensitivities"],
        activityPreferences=["Loves active & energetic play"],
    )
    normalized_req = ActivityRequest(
        user_profile=req.user_profile or default_profile,
    )

    if not GEMINI_API_KEY or not genai:
        logging.warning({
            "event": "gemini_not_configured",
            "gemini_import_ok": bool(genai),
            "gemini_key_present": bool(GEMINI_API_KEY),
        })
        raise HTTPException(status_code=503, detail="Gemini not configured")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = _build_activity_prompt(normalized_req, obs, aqi_info)
        logging.info({
            "event": "gemini_request",
            "model": "gemini-2.5-flash-lite",
            "pm25": pm25_val,
            "aqi": aqi_info.get("aqi"),
            "level": aqi_info.get("level"),
        })
        resp = model.generate_content(prompt)
        parsed = _parse_gemini_json(getattr(resp, "text", "") or "")
        if not parsed:
            logging.warning({"event": "gemini_non_json_output"})
            raise HTTPException(status_code=502, detail="Model returned non-JSON output")

        parsed.setdefault("recommendation_level", aqi_info.get("level") or "Moderate Caution")
        ra = parsed.setdefault("recommended_activity", {})
        ra.setdefault("name", "Family-Friendly Activity")
        ra.setdefault("location_name", "Malang, Indonesia")
        ra.setdefault("developmental_benefit", "Supports age-appropriate development.")
        ra.setdefault("safety_tip", "Follow local air quality guidance.")
        try:
            if pm25_val is not None:
                parsed["current_aqi"] = int(round(float(pm25_val)))
            else:
                parsed["current_aqi"] = int(parsed.get("current_aqi") or (aqi_info.get("aqi") or 75))
        except Exception:
            parsed["current_aqi"] = int(aqi_info.get("aqi") or 75)

        logging.info({
            "event": "recommendation_success",
            "current_aqi_reported": parsed.get("current_aqi"),
            "recommendation_level": parsed.get("recommendation_level"),
        })
        return parsed
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("gemini_call_failed")
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {str(e)}")