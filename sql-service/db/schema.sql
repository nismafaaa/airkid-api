-- Hourly observations (processed)
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
  timezone TEXT NULL,              -- time.tz
  city_url TEXT NULL,              -- data.city.url
  raw_json JSONB NULL,             -- entire original payload
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (city, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_aqi_observations_city_time
  ON aqi_observations (city, observed_at DESC);

-- Daily forecast (pm10, pm25, uvi arrays)
CREATE TABLE IF NOT EXISTS aqi_forecast_daily (
  id BIGSERIAL PRIMARY KEY,
  city TEXT NOT NULL,
  pollutant TEXT NOT NULL,  -- pm10 | pm25 | uvi
  day DATE NOT NULL,        -- forecast day
  avg INTEGER NULL,
  min INTEGER NULL,
  max INTEGER NULL,
  source_idx INTEGER NULL,  -- "idx" from payload
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (city, pollutant, day)
);

CREATE INDEX IF NOT EXISTS idx_aqi_forecast_daily_city_day
  ON aqi_forecast_daily (city, day DESC);
