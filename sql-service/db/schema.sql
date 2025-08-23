CREATE TABLE IF NOT EXISTS aqi_observations (
  id BIGSERIAL PRIMARY KEY,
  city TEXT NOT NULL,
  lat DOUBLE PRECISION NULL,
  lon DOUBLE PRECISION NULL,
  source_idx INTEGER NULL,         -- "idx" from payload
  dominant_pol TEXT NULL,          -- "dominentpol"
  aqi INTEGER NULL,                -- headline AQI
  pm25 DOUBLE PRECISION NULL,      -- iaqi.pm25.v
  dew DOUBLE PRECISION NULL,       -- iaqi.dew.v
  humidity DOUBLE PRECISION NULL,  -- iaqi.h.v
  pressure DOUBLE PRECISION NULL,  -- iaqi.p.v
  rainfall DOUBLE PRECISION NULL,  -- iaqi.r.v
  temperature DOUBLE PRECISION NULL, -- iaqi.t.v
  wind DOUBLE PRECISION NULL,      -- iaqi.w.v
  observed_at TIMESTAMPTZ NOT NULL, -- time.iso (stored as UTC)
  observed_date DATE GENERATED ALWAYS AS (observed_at::date) STORED,
  timezone TEXT NULL,              -- time.tz
  city_url TEXT NULL,              -- data.city.url
  raw_json JSONB NULL,             -- entire original payload
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90)),
  CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180)),
  CHECK (aqi IS NULL OR (aqi BETWEEN 0 AND 500)),
  CHECK (pm25 IS NULL OR (pm25 >= 0)),
  CHECK (humidity IS NULL OR (humidity BETWEEN 0 AND 100)),
  CHECK (pressure IS NULL OR (pressure > 0)),
  CHECK (rainfall IS NULL OR (rainfall >= 0)),
  CHECK (wind IS NULL OR (wind >= 0)),
  UNIQUE (city, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_aqi_observations_city_time
  ON aqi_observations (city, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_aqi_observations_observed_at
  ON aqi_observations (observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_aqi_observations_city_day
  ON aqi_observations (city, observed_date DESC);

CREATE INDEX IF NOT EXISTS idx_aqi_observations_raw_json_gin
  ON aqi_observations USING GIN (raw_json);

CREATE TABLE IF NOT EXISTS aqi_forecast_daily (
  id BIGSERIAL PRIMARY KEY,
  city TEXT NOT NULL,
  pollutant TEXT NOT NULL,  
  day DATE NOT NULL,        
  avg INTEGER NULL,
  min INTEGER NULL,
  max INTEGER NULL,
  source_idx INTEGER NULL,  
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (pollutant IN ('pm10','pm25','uvi')),
  CHECK (min IS NULL OR min >= 0),
  CHECK (avg IS NULL OR avg >= 0),
  CHECK (max IS NULL OR max >= 0),
  CHECK (min IS NULL OR max IS NULL OR min <= max),
  CHECK (avg IS NULL OR min IS NULL OR avg >= min),
  CHECK (avg IS NULL OR max IS NULL OR avg <= max),
  UNIQUE (city, pollutant, day)
);

ALTER TABLE aqi_forecast_daily
  ADD COLUMN IF NOT EXISTS source TEXT;

UPDATE aqi_forecast_daily
SET source = 'provider'
WHERE source IS NULL;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'aqi_forecast_daily'::regclass
      AND contype = 'u'
      AND conname = 'aqi_forecast_daily_city_pollutant_day_key'
  ) THEN
    ALTER TABLE aqi_forecast_daily
      DROP CONSTRAINT aqi_forecast_daily_city_pollutant_day_key;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'aqi_forecast_daily'::regclass
      AND contype = 'u'
      AND conname = 'aqi_forecast_daily_city_pollutant_day_source_key'
  ) THEN
    ALTER TABLE aqi_forecast_daily
      ADD CONSTRAINT aqi_forecast_daily_city_pollutant_day_source_key
      UNIQUE (city, pollutant, day, source);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_day
  ON aqi_forecast_daily (city, day DESC);

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_pollutant_created
  ON aqi_forecast_daily (city, pollutant, created_at DESC, day);

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_pollutant_sourceidx
  ON aqi_forecast_daily (city, pollutant, source_idx DESC, day);

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_pollutant_source_created
  ON aqi_forecast_daily (city, pollutant, source, created_at DESC, day);

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_pollutant_source_sourceidx
  ON aqi_forecast_daily (city, pollutant, source, source_idx DESC, day);
