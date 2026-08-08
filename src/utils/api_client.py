"""
Open-Meteo API client for weather and air quality data.

Supports two modes:
  * Historical backfill (archive endpoints) -> used by data ingestion (Stage 1)
  * Live forecast (forecast endpoints)       -> used by inference / serving

Field names are returned VERBATIM from Open-Meteo (raw schema), which is the
schema the training pipeline and model consume (see models/features.txt).
"""

import time
import random
import requests
from typing import Optional, Dict, List

from .logger import get_logger


logger = get_logger(__name__)


class OpenMeteoError(RuntimeError):
    """Raised when the Open-Meteo API fails after all retries.

    Ingestion is configured to halt-and-report, so this propagates up rather
    than being silently swallowed, preventing a partial/garbage dataset.
    """


class OpenMeteoClient:
    """
    Client for the Open-Meteo API (free weather + air-quality data).

    Historical archive endpoints are used for the 2022-today backfill; the
    live forecast endpoints are kept for the inference/serving path.
    """

    # Live forecast endpoints
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # Historical archive endpoints
    ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
    # The air-quality API serves historical data too (start_date/end_date),
    # with coverage back to ~2022-07-29.
    ARCHIVE_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    TIMEZONE = "Asia/Kolkata"

    # Exact hourly fields that produce the 31-feature raw schema the model uses.
    HISTORICAL_WEATHER_HOURLY: List[str] = [
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "is_day",
    ]

    HISTORICAL_AQ_HOURLY: List[str] = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust",
        "aerosol_optical_depth",
        "us_aqi",  # prediction target
    ]

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        rate_limit_min: float = 1.0,
        rate_limit_max: float = 2.0,
    ):
        """
        Args:
            timeout: Per-request timeout in seconds.
            max_retries: Number of retries on transient failure before giving up.
            retry_delay: Base delay (seconds) for exponential backoff.
            rate_limit_min/max: Random sleep window between successful calls to
                stay under Open-Meteo's fair-use rate limits.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_min = rate_limit_min
        self.rate_limit_max = rate_limit_max

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AQI-Prediction-MLOps/1.0"})

    # ------------------------------------------------------------------ #
    # Internal request helper (retries + backoff + rate limiting)
    # ------------------------------------------------------------------ #
    def _request(self, url: str, params: Dict, what: str) -> Dict:
        """
        Perform a GET with retries and exponential backoff.

        Raises OpenMeteoError if all retries are exhausted (halt-and-report).
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Explicit handling for rate limiting
                if response.status_code == 429:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"    Rate limited (429) on {what}; waiting {wait:.0f}s")
                    time.sleep(wait)
                    last_error = OpenMeteoError("HTTP 429 rate limited")
                    continue

                response.raise_for_status()
                data = response.json()

                # API-level error payloads use {"error": true, "reason": "..."}
                if isinstance(data, dict) and data.get("error"):
                    raise OpenMeteoError(f"API error for {what}: {data.get('reason')}")

                # Polite pacing between successful calls
                time.sleep(random.uniform(self.rate_limit_min, self.rate_limit_max))
                return data

            except (requests.exceptions.RequestException, ValueError, OpenMeteoError) as e:
                last_error = e
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"    {what} attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(wait)

        raise OpenMeteoError(f"{what} failed after {self.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------ #
    # Historical (archive) fetch -- used by Stage 1 data ingestion
    # ------------------------------------------------------------------ #
    def fetch_historical_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Optional[Dict]:
        """
        Fetch historical hourly weather from the Open-Meteo archive (ERA5).

        Returns the raw JSON dict (with an "hourly" dict of equal-length lists,
        including "time"), or raises OpenMeteoError on definitive failure.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HISTORICAL_WEATHER_HOURLY),
            "timezone": self.TIMEZONE,
        }
        return self._request(
            self.ARCHIVE_WEATHER_URL, params, f"weather {start_date}->{end_date}"
        )

    def fetch_air_quality(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Optional[Dict]:
        """
        Fetch historical hourly air quality (incl. us_aqi target) for a date range.

        Returns the raw JSON dict (with an "hourly" dict of equal-length lists,
        including "time"), or raises OpenMeteoError on definitive failure.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HISTORICAL_AQ_HOURLY),
            "timezone": self.TIMEZONE,
        }
        return self._request(
            self.ARCHIVE_AQ_URL, params, f"air-quality {start_date}->{end_date}"
        )

    # ------------------------------------------------------------------ #
    # Live forecast fetch -- used by the inference / serving path
    # ------------------------------------------------------------------ #
    def fetch_weather_forecast(
        self, lat: float, lon: float, forecast_days: int = 2
    ) -> Optional[Dict]:
        """Fetch weather forecast (serving path). Returns None on failure."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "relative_humidity_2m",
                    "dew_point_2m",
                    "precipitation",
                    "pressure_msl",
                    "surface_pressure",
                    "cloud_cover",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "wind_direction_10m",
                    "is_day",
                ],
                "forecast_days": min(forecast_days, 3),
                "timezone": self.TIMEZONE,
            }
            response = self.session.get(self.FORECAST_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Fallback: some models omit gusts -> approximate with wind speed
            if "hourly" in data and "wind_gusts_10m" not in data["hourly"]:
                if "wind_speed_10m" in data["hourly"]:
                    data["hourly"]["wind_gusts_10m"] = data["hourly"]["wind_speed_10m"]
            return data
        except Exception as e:
            logger.error(f"Failed to fetch weather forecast: {e}")
            return None

    def fetch_air_quality_forecast(
        self, lat: float, lon: float, forecast_days: int = 2
    ) -> Optional[Dict]:
        """Fetch air-quality forecast (serving path). Returns None on failure."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "pm2_5",
                    "pm10",
                    "carbon_monoxide",
                    "nitrogen_dioxide",
                    "sulphur_dioxide",
                    "ozone",
                    "dust",
                    "aerosol_optical_depth",
                    "us_aqi",
                ],
                "forecast_days": min(forecast_days, 3),
                "timezone": self.TIMEZONE,
            }
            response = self.session.get(self.AQ_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch AQ forecast: {e}")
            return None
