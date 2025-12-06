# src/components/data_ingestion.py
import sys
import json
import time
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    raw_data_dir: Path
    cities_config_path: Path = Path("configs/cities.yaml")
    start_date: str = "2022-08-05"
    end_date: Optional[str] = None
    timeout: int = 120
    retry_attempts: int = 5
    retry_delay: int = 10
    requests_per_minute: int = 15
    checkpoint_file: str = "ingestion_checkpoint.json"
    save_after_each_city: bool = True

@dataclass
class DataIngestionArtifact:
    raw_data_path: Path
    cities_processed: int
    cities_failed: List[str]
    total_records: int
    date_range: str
    success: bool
    message: str

class CheckpointManager:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_cities": [],
            "failed_cities": {},
            "last_updated": None,
            "total_records": 0,
            "start_date": None,
            "end_date": None,
        }

    def save(self):
        self._data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def set_date_range(self, start: str, end: str):
        self._data["start_date"] = start
        self._data["end_date"] = end
        self.save()

    def mark_completed(self, city: str, records: int):
        if city not in self._data["completed_cities"]:
            self._data["completed_cities"].append(city)
        self._data["total_records"] += records
        self._data["failed_cities"].pop(city, None)
        self.save()

    def mark_failed(self, city: str, error: str):
        self._data["failed_cities"][city] = {"error": error, "timestamp": datetime.now().isoformat()}
        self.save()

    def get_pending(self, all_cities: List[str]) -> List[str]:
        return [c for c in all_cities if c not in self._data["completed_cities"]]

    def reset(self):
        self._data = {
            "completed_cities": [],
            "failed_cities": {},
            "last_updated": None,
            "total_records": 0,
            "start_date": None,
            "end_date": None,
        }
        self.save()
        logger.info("Checkpoint reset")

    @property
    def completed(self):
        return self._data["completed_cities"]

class RateLimitedClient:
    def __init__(self, requests_per_minute: int = 15):
        self.interval = 60.0 / requests_per_minute
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        sleep_time = self.interval - (now - self.last_call)
        if sleep_time > 0:
            time.sleep(sleep_time + random.uniform(0.1, 0.5))
        self.last_call = time.time()

class DataIngestion:
    WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,pressure_msl,cloud_cover,wind_gusts_10m"
    AQ_PARAMS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth"

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        if self.config.end_date is None:
            self.config.end_date = datetime.now().strftime("%Y-%m-%d")

        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = CheckpointManager(self.config.raw_data_dir / self.config.checkpoint_file)
        self.client = RateLimitedClient(self.config.requests_per_minute)
        self.cities = self._load_cities()

    def _load_cities(self) -> Dict[str, Dict]:
        with open(self.config.cities_config_path) as f:
            data = yaml.safe_load(f)
        return data["cities"]

    def _fetch(self, url: str, params: dict) -> Optional[dict]:
        for attempt in range(self.config.retry_attempts):
            try:
                self.client.wait()
                response = requests.get(url, params=params, timeout=self.config.timeout)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"HTTP {response.status_code}. Retrying...")
                    time.sleep(self.config.retry_delay)
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                time.sleep(self.config.retry_delay)
        return None

    def _fetch_city_data(self, city: str, info: dict) -> Optional[pd.DataFrame]:
        params = {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "hourly": self.WEATHER_PARAMS,
            "timezone": "Asia/Kolkata",
        }
        weather = self._fetch(self.WEATHER_URL, params)
        params["hourly"] = self.AQ_PARAMS
        aq = self._fetch(self.AQ_URL, params)

        if not weather or "hourly" not in weather or not aq or "hourly" not in aq:
            return None

        df = pd.DataFrame({
            "datetime": pd.to_datetime(weather["hourly"]["time"]),
            "city": city,
            "state": info["state"],
            "latitude": info["lat"],
            "longitude": info["lon"],
            **{k: weather["hourly"].get(k) for k in self.WEATHER_PARAMS.split(",")},
            **{k: aq["hourly"].get(k) for k in self.AQ_PARAMS.split(",")},
        })
        return df.dropna(subset=["datetime"])

    def initiate_data_ingestion(self, resume: bool = True) -> DataIngestionArtifact:
        logger.info("STARTING DATA INGESTION")
        all_cities = list(self.cities.keys())
        pending = self.checkpoint.get_pending(all_cities) if resume else all_cities

        if not resume:
            self.checkpoint.reset()
            shutil.rmtree(self.config.raw_data_dir / "cities", ignore_errors=True)

        self.checkpoint.set_date_range(self.config.start_date, self.config.end_date)

        failed = []
        dfs = []

        with tqdm(pending, desc="Cities") as pbar:
            for city in pbar:
                pbar.set_postfix({"city": city[:15]})
                try:
                    df = self._fetch_city_data(city, self.cities[city])
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                        if self.config.save_after_each_city:
                            safe_name = city.lower().replace(" ", "_")
                            path = self.config.raw_data_dir / "cities" / f"{safe_name}.parquet"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            df.to_parquet(path, index=False)
                        self.checkpoint.mark_completed(city, len(df))
                    else:
                        failed.append(city)
                        self.checkpoint.mark_failed(city, "No data")
                except Exception as e:
                    failed.append(city)
                    self.checkpoint.mark_failed(city, str(e))

        final_path = self.config.raw_data_dir / "aqi_india_raw.parquet"
        if dfs:
            pd.concat(dfs, ignore_index=True).to_parquet(final_path, index=False)

        artifact = DataIngestionArtifact(
            raw_data_path=final_path,
            cities_processed=len(all_cities) - len(failed),
            cities_failed=failed,
            total_records=sum(len(df) for df in dfs) if dfs else 0,
            date_range=f"{self.config.start_date} to {self.config.end_date}",
            success=len(failed) < len(all_cities),
            message=f"Processed {len(all_cities)-len(failed)}/{len(all_cities)} cities"
        )
        logger.info(artifact.message)
        return artifact