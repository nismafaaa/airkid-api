import os
import json
from typing import Any, Dict, List
from datetime import datetime

import pytz
import functions_framework
from dateutil import parser as dtparser
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine, text

# ---------- DB engine (lazy init) ----------
_connector = None
_engine_instance = None

def _getconn():
    ip_type = IPTypes.PRIVATE if os.getenv("DB_PRIVATE_IP", "false").lower() == "true" else IPTypes.PUBLIC
    global _connector
    if _connector is None:
        _connector = Connector()
    return _connector.connect(
        os.environ["INSTANCE_CONNECTION_NAME"],
        "pg8000",
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        db=os.environ["DB_NAME"],
        ip_type=ip_type,
    )

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = create_engine(
            "postgresql+pg8000://",
            creator=_getconn,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
        )
    return _engine_instance

storage_client = storage.Client()

# ---------- Helpers ----------
def _to_float(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def _parse_observed_at(iso_or_s: str) -> datetime:
    if not iso_or_s:
        return None
    dt = dtparser.isoparse(iso_or_s)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(pytz.UTC)

def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    d = payload.get("data", {}) or {}
    city_obj = d.get("city", {}) or {}
    time_obj = d.get("time", {}) or {}
    iaqi = d.get("iaqi", {}) or {}

    def iv(key):
        node = iaqi.get(key) or {}
        return _to_float(node.get("v"))

    geo = city_obj.get("geo") or []
    lat = _to_float(geo[0]) if len(geo) >= 1 else None
    lon = _to_float(geo[1]) if len(geo) >= 2 else None

    iso = time_obj.get("iso") or time_obj.get("s")
    observed_at = _parse_observed_at(iso)

    return {
        "city": city_obj.get("name"),
        "lat": lat,
        "lon": lon,
        "source_idx": d.get("idx"),
        "dominant_pol": d.get("dominentpol"),
        "aqi": d.get("aqi"),
        "pm25": iv("pm25"),
        "dew": iv("dew"),
        "humidity": iv("h"),
        "pressure": iv("p"),
        "rainfall": iv("r"),
        "temperature": iv("t"),
        "wind": iv("w"),
        "observed_at": observed_at,
        "timezone": time_obj.get("tz"),
        "city_url": city_obj.get("url"),
        "raw_json": json.dumps(payload),
    }

def _extract_forecasts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    d = payload.get("data", {}) or {}
    city_obj = d.get("city", {}) or {}
    city = city_obj.get("name")
    source_idx = d.get("idx")
    out = []
    daily = ((d.get("forecast") or {}).get("daily")) or {}
    for pollutant, rows in daily.items():
        for r in rows or []:
            try:
                day = dtparser.isoparse(str(r.get("day"))).date()
            except Exception:
                continue
            out.append({
                "city": city,
                "pollutant": pollutant,
                "day": day,
                "avg": r.get("avg"),
                "min": r.get("min"),
                "max": r.get("max"),
                "source_idx": source_idx,
            })
    return out

@functions_framework.cloud_event
def gcs_to_postgres(cloud_event):
    data = getattr(cloud_event, "data", {}) or (cloud_event.get("data") if isinstance(cloud_event, dict) else {})
    name = (data or {}).get("name")
    bucket = (data or {}).get("bucket")
    if not bucket or not name:
        print("Missing bucket/name in event; skipping")
        return

    if not name.startswith("aqi_data/"):
        print(f"Object not in aqi_data/ prefix: {name}; skipping")
        return

    try:
        blob = storage_client.bucket(bucket).blob(name)
        raw = blob.download_as_text()
        payload = json.loads(raw)
    except Exception as e:
        print(f"Failed to read/parse GCS object {bucket}/{name}: {e}")
        return

    if payload.get("status") != "ok" or "data" not in payload:
        print(f"Payload not OK or missing data: {bucket}/{name}")
        return

    obs = _extract_observation(payload)
    if not obs.get("city") or not obs.get("observed_at"):
        print(f"Missing city/observed_at; skipping: {bucket}/{name}")
        return

    try:
        with get_engine().begin() as conn:
            conn.execute(
                text("""
                INSERT INTO aqi_observations
                (city, lat, lon, source_idx, dominant_pol, aqi, pm25, dew, humidity, pressure, rainfall, temperature, wind, observed_at, timezone, city_url, raw_json)
                VALUES
                (:city, :lat, :lon, :source_idx, :dominant_pol, :aqi, :pm25, :dew, :humidity, :pressure, :rainfall, :temperature, :wind, :observed_at, :timezone, :city_url, CAST(:raw_json AS JSONB))
                ON CONFLICT (city, observed_at) DO UPDATE SET
                  lat=EXCLUDED.lat,
                  lon=EXCLUDED.lon,
                  source_idx=EXCLUDED.source_idx,
                  dominant_pol=EXCLUDED.dominant_pol,
                  aqi=EXCLUDED.aqi,
                  pm25=EXCLUDED.pm25,
                  dew=EXCLUDED.dew,
                  humidity=EXCLUDED.humidity,
                  pressure=EXCLUDED.pressure,
                  rainfall=EXCLUDED.rainfall,
                  temperature=EXCLUDED.temperature,
                  wind=EXCLUDED.wind,
                  timezone=EXCLUDED.timezone,
                  city_url=EXCLUDED.city_url,
                  raw_json=EXCLUDED.raw_json
                """),
                obs,
            )

            forecasts = _extract_forecasts(payload)
            if forecasts:
                conn.execute(
                    text("""
                    INSERT INTO aqi_forecast_daily
                    (city, pollutant, day, avg, min, max, source_idx)
                    VALUES
                    (:city, :pollutant, :day, :avg, :min, :max, :source_idx)
                    ON CONFLICT (city, pollutant, day) DO UPDATE SET
                      avg=EXCLUDED.avg,
                      min=EXCLUDED.min,
                      max=EXCLUDED.max,
                      source_idx=EXCLUDED.source_idx
                    """),
                    forecasts,
                )

        print(f"Upserted observation and {len(forecasts or [])} forecast rows from {bucket}/{name}")
    except Exception as e:
        print(f"DB upsert failed for {bucket}/{name}: {e}")