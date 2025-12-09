"""
Stage 1: Data Ingestion
Download historical weather and AQ data from Open-Meteo API
Exact logic from notebook: 01_data_ingestion.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader, load_cities_config
from ..utils.api_client import OpenMeteoClient


logger = get_logger(__name__)


class DataIngestion:
    """
    Download and combine weather + AQ data for all cities
    Matches notebook ingestion logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize data ingestion
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get ingestion parameters
        self.start_date = config.get("data_ingestion.start_date", "2022-08-05")
        self.end_date = config.get("data_ingestion.end_date") or datetime.now().strftime("%Y-%m-%d")
        
        # API settings
        api_config = config.get_section("data_ingestion").get("api", {})
        self.api_client = OpenMeteoClient(
            timeout=api_config.get("timeout", 120),
            max_retries=api_config.get("retry_attempts", 3),
            retry_delay=api_config.get("retry_delay", 5),
            rate_limit_min=api_config.get("rate_limit_min", 1.0),
            rate_limit_max=api_config.get("rate_limit_max", 2.0)
        )
        
        # Output settings
        output_config = config.get_section("data_ingestion").get("output", {})
        self.raw_dir = Path(output_config.get("raw_dir", "data/raw"))
        self.cities_per_batch = output_config.get("cities_per_batch", 3)
        
        # Create output directory
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cities
        self.cities = load_cities_config()
        
        logger.info(f"Data Ingestion initialized")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Cities: {len(self.cities)}")
    
    def _download_city_data(self, city_name: str, city_info: Dict) -> Optional[pd.DataFrame]:
        """
        Download weather + AQ data for one city
        Matches notebook ingestion logic
        """
        lat = city_info["lat"]
        lon = city_info["lon"]
        state = city_info["state"]

        logger.info(f"Processing: {city_name}, {state}")
        logger.info(f"Coordinates: ({lat}, {lon})")

        all_weather = []
        all_aq = []

        # Download year-by-year
        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        total_years = end_year - start_year + 1
        logger.info(f"  Will download {total_years} years: {start_year} to {end_year}")

        for year_index, year in enumerate(range(start_year, end_year + 1), start=1):
            year_start = self.start_date if year == start_year else f"{year}-01-01"
            year_end = self.end_date if year == end_year else f"{year}-12-31"

            logger.info(f"  [{year_index}/{total_years}] Downloading {year}: {year_start} to {year_end}")

            # Weather Data
            logger.info(f"    Fetching weather…")
            weather_data = self.api_client.fetch_historical_weather(lat, lon, year_start, year_end)

            if weather_data and "hourly" in weather_data:
                weather_df = pd.DataFrame(weather_data["hourly"])
                all_weather.append(weather_df)
                logger.info(f"    OK Weather: {len(weather_df):,} records")
            else:
                logger.warning(f"    FAILED Weather for {year}")

            time.sleep(1)

            # AQ Data
            logger.info(f"    Fetching air quality…")
            aq_data = self.api_client.fetch_air_quality(lat, lon, year_start, year_end)

            if aq_data and "hourly" in aq_data:
                aq_df = pd.DataFrame(aq_data["hourly"])
                all_aq.append(aq_df)
                logger.info(f"    OK AQ: {len(aq_df):,} records")
            else:
                logger.warning(f"    FAILED AQ for {year}")

            logger.info(f"    Completed year {year} ({year_index}/{total_years})")
            time.sleep(1)

        if not all_weather:
            logger.error(f"No weather data for {city_name}")
            return None

        weather_combined = pd.concat(all_weather, ignore_index=True).drop_duplicates(subset=["time"])
        weather_combined["time"] = pd.to_datetime(weather_combined["time"])

        if all_aq:
            aq_combined = pd.concat(all_aq, ignore_index=True).drop_duplicates(subset=["time"])
            aq_combined["time"] = pd.to_datetime(aq_combined["time"])
            merged = pd.merge(weather_combined, aq_combined, on="time", how="left")
        else:
            logger.warning(f"No AQ data for {city_name}, using weather only")
            merged = weather_combined.copy()

        merged.insert(0, "city", city_name)
        merged.insert(1, "state", state)
        merged.insert(2, "latitude", lat)
        merged.insert(3, "longitude", lon)
        merged.rename(columns={"time": "datetime"}, inplace=True)

        logger.info(f"OK {city_name} complete: {len(merged):,} records")

        return merged
    
    def run(self) -> str:
        """
        Run data ingestion for all cities
        
        Returns:
            Path to output parquet file
        """
        logger.info("="*90)
        logger.info("STARTING DATA INGESTION")
        logger.info("="*90)
        
        all_data = []
        failed_cities = []
        
        for idx, (city_name, city_info) in enumerate(self.cities.items(), 1):
            logger.info(f"\n[{idx}/{len(self.cities)}] {city_name}")
            logger.info("-" * 70)
            
            try:
                city_df = self._download_city_data(city_name, city_info)
                
                if city_df is not None:
                    all_data.append(city_df)
                else:
                    failed_cities.append(city_name)
                
                # Batch cooldown (from notebook)
                if idx % self.cities_per_batch == 0 and idx < len(self.cities):
                    logger.info("\nCooldown API cooldown (90 seconds)...")
                    time.sleep(90)
                else:
                    time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error processing {city_name}: {e}")
                failed_cities.append(city_name)
                time.sleep(30)
        
        # Combine all cities
        if not all_data:
            raise RuntimeError("No data downloaded for any city!")
        
        logger.info("\n" + "="*90)
        logger.info("COMBINING DATA FROM ALL CITIES")
        logger.info("="*90)
        
        final_df = pd.concat(all_data, ignore_index=True)
        final_df["datetime"] = pd.to_datetime(final_df["datetime"])
        
        # Sort by city and datetime
        final_df = final_df.sort_values(["city", "datetime"]).reset_index(drop=True)
        
        # Remove duplicates (from notebook)
        final_df = final_df.drop_duplicates(subset=["city", "datetime"], keep="last")
        
        # Save to parquet
        output_file = self.raw_dir / "aqi_india_raw.parquet"
        final_df.to_parquet(output_file, index=False)
        
        # Generate metrics
        metrics = self._generate_metrics(final_df, failed_cities)
        
        # Save metrics
        metrics_file = self.raw_dir / "ingestion_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        self._print_summary(final_df, failed_cities)
        
        logger.info(f"\nOK Data ingestion complete!")
        logger.info(f"Output: {output_file}")
        logger.info(f"Metrics: {metrics_file}")
        
        return str(output_file)
    
    def _generate_metrics(self, df: pd.DataFrame, failed_cities: List[str]) -> Dict:
        """Generate ingestion metrics"""
        total_cities = len(self.cities)
        successful_cities = total_cities - len(failed_cities)
        
        metrics = {
            "total_cities": total_cities,
            "successful_cities": successful_cities,
            "failed_cities": len(failed_cities),
            "failed_city_names": failed_cities,
            "total_records": int(len(df)),
            "date_range_start": str(df["datetime"].min()),
            "date_range_end": str(df["datetime"].max()),
            "total_columns": int(len(df.columns)),
            "cities_in_data": df["city"].nunique(),
            "pm25_coverage_pct": float(df["pm2_5"].notna().sum() / len(df) * 100),
            "us_aqi_coverage_pct": float(df["us_aqi"].notna().sum() / len(df) * 100),
        }
        
        return metrics
    
    def _print_summary(self, df: pd.DataFrame, failed_cities: List[str]):
        """Print ingestion summary"""
        print("\n" + "="*90)
        print("DATA INGESTION SUMMARY")
        print("="*90)
        print(f"Total Records:        {len(df):,}")
        print(f"Total Columns:        {len(df.columns)}")
        print(f"Cities:               {df['city'].nunique()}/{len(self.cities)}")
        print(f"Date Range:           {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"\nData Coverage:")
        print(f"  PM2.5:              {df['pm2_5'].notna().sum() / len(df) * 100:.2f}%")
        print(f"  PM10:               {df['pm10'].notna().sum() / len(df) * 100:.2f}%")
        print(f"  US AQI:             {df['us_aqi'].notna().sum() / len(df) * 100:.2f}%")
        
        if failed_cities:
            print(f"\nWarning  Failed Cities ({len(failed_cities)}):")
            for city in failed_cities:
                print(f"  - {city}")
        
        print("="*90)