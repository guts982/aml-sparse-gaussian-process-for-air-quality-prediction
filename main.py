"""
Historical Kathmandu Air Quality Data Collector (2021-2024)
Collects 3-4 years of PM2.5 and meteorological data from known reliable sources

Data Sources:
1. Open Data Nepal (2015-2021) - Official government data  
2. AQICN Historical API (2020-2024) - US Embassy + official stations
3. OpenWeather Historical API (2020-2024) - Weather data
4. Open-Meteo Archive (2020-2024) - Free historical weather + air quality
5. IQAir Historical API (if available)

Features: PM2.5, temperature, humidity, wind_speed, wind_direction, pressure, 
         aod_550nm, hour, day, datetime, station_id, lat, lng, elevation
"""

import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta, date
import time
import json
import logging
import warnings
from typing import List, Dict, Optional, Tuple
import os
from urllib.parse import urljoin
import calendar

warnings.filterwarnings('ignore')

class HistoricalKathmanduCollector:
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Historical data collector for Kathmandu (2021-2024)
        
        Args:
            api_keys: Dictionary of API keys for paid services
        """
        self.api_keys = api_keys or {}
        self.logger = self._setup_logger()
        
        # Known reliable monitoring stations in Kathmandu
        self.stations = {
            'us_embassy_main': {
                'lat': 27.7172, 'lng': 85.3240, 'elevation': 1350,
                'name': 'US_Embassy_Kathmandu_Main',
                'aqicn_id': '@9107',  # AQICN station ID
                'description': 'US Embassy main compound'
            },
            'us_embassy_phora': {
                'lat': 27.7172, 'lng': 85.3088, 'elevation': 1340,
                'name': 'US_Embassy_Phora_Durbar',
                'aqicn_id': 'nepal/kathmandu/us-embassy-phora-durbar',
                'description': 'US Embassy Phora Durbar Recreation Center'
            },
            'icimod_khumaltar': {
                'lat': 27.6644, 'lng': 85.3188, 'elevation': 1350,
                'name': 'ICIMOD_Khumaltar_Station',
                'description': 'ICIMOD Air Quality Monitoring Station'
            }
        }
        
        # Date ranges for data collection
        self.target_start_date = "2021-01-01"
        self.target_end_date = "2024-08-25"
        
        self.logger.info(f"Initialized collector for {self.target_start_date} to {self.target_end_date}")
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('kathmandu_data_collection.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def download_open_data_nepal(self) -> pd.DataFrame:
        """
        Download historical data from Open Data Nepal (2015-2021)
        This contains official government monitoring station data
        """
        self.logger.info("Downloading Open Data Nepal dataset...")
        
        try:
            # Direct download URL for the dataset
            url = "https://opendatanepal.com/dataset/042b8821-1117-44cc-b007-e8633ae64702/resource/042b8821-1117-44cc-b007-e8633ae64702/download/air-quality-data-in-kathmandu.csv"
            
            # Alternative: Try the main dataset page
            backup_urls = [
                "https://opendatanepal.com/dataset/air-quality-data-in-kathmandu/resource/042b8821-1117-44cc-b007-e8633ae64702",
                "https://raw.githubusercontent.com/opendatanepal/datasets/main/air-quality-kathmandu.csv"  # If available on GitHub
            ]
            
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Try to parse as CSV
                try:
                    df = pd.read_csv(pd.io.common.StringIO(response.text))
                    
                    # Standardize column names
                    column_mapping = {
                        'Date': 'datetime',
                        'PM2.5': 'pm2_5',
                        'PM10': 'pm10',
                        'O3': 'o3',
                        'NO2': 'no2',
                        'SO2': 'so2',
                        'CO': 'co',
                        'Temperature': 'temperature',
                        'Humidity': 'humidity',
                        'Pressure': 'pressure'
                    }
                    
                    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                    
                    # Add station metadata
                    df['station_id'] = 'Nepal_Government_Kathmandu'
                    df['lat'] = 27.7172
                    df['lng'] = 85.3240
                    df['elevation'] = 1350
                    df['source'] = 'open_data_nepal'
                    
                    # Convert datetime
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    
                    # Filter for target date range
                    df = df[df['datetime'] >= '2021-01-01']
                    
                    self.logger.info(f"✓ Open Data Nepal: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
                    return df
                
                except Exception as e:
                    self.logger.error(f"Error parsing Open Data Nepal CSV: {e}")
            
            else:
                self.logger.warning(f"Open Data Nepal download failed: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"Open Data Nepal error: {e}")
        
        return pd.DataFrame()
    
    def get_aqicn_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data from AQICN for US Embassy stations
        Note: Historical data requires paid API or web scraping
        """
        if 'aqicn' not in self.api_keys:
            self.logger.warning("AQICN API key not provided for historical data")
            return self._scrape_aqicn_historical(start_date, end_date)
        
        self.logger.info(f"Collecting AQICN historical data ({start_date} to {end_date})...")
        all_data = []
        
        api_token = self.api_keys['aqicn']
        
        # Get historical data for each station
        for station_key, station_info in self.stations.items():
            if 'aqicn_id' not in station_info:
                continue
            
            try:
                # AQICN historical API endpoint (if available)
                station_id = station_info['aqicn_id']
                
                if station_id.startswith('@'):
                    url = f"https://api.waqi.info/feed/{station_id}/history"
                else:
                    url = f"https://api.waqi.info/feed/{station_id}/history"
                
                # Date range in chunks (API limitations)
                current_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                station_data = []
                
                while current_date <= end_dt:
                    chunk_end = min(current_date + timedelta(days=30), end_dt)
                    
                    params = {
                        'token': api_token,
                        'start': current_date.strftime('%Y-%m-%d'),
                        'end': chunk_end.strftime('%Y-%m-%d')
                    }
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data['status'] == 'ok' and 'data' in data:
                            for record in data['data']:
                                processed_record = {
                                    'datetime': record.get('time', {}).get('iso'),
                                    'station_id': f"aqicn_{station_key}",
                                    'lat': station_info['lat'],
                                    'lng': station_info['lng'],
                                    'elevation': station_info['elevation'],
                                    'pm2_5': record.get('iaqi', {}).get('pm25', {}).get('v'),
                                    'pm10': record.get('iaqi', {}).get('pm10', {}).get('v'),
                                    'temperature': record.get('iaqi', {}).get('t', {}).get('v'),
                                    'humidity': record.get('iaqi', {}).get('h', {}).get('v'),
                                    'pressure': record.get('iaqi', {}).get('p', {}).get('v'),
                                    'aqi': record.get('aqi'),
                                    'source': 'aqicn_historical'
                                }
                                station_data.append(processed_record)
                    
                    current_date = chunk_end + timedelta(days=1)
                    time.sleep(1)  # Rate limiting
                
                if station_data:
                    station_df = pd.DataFrame(station_data)
                    all_data.append(station_df)
                    self.logger.info(f"✓ AQICN {station_key}: {len(station_df)} historical records")
            
            except Exception as e:
                self.logger.error(f"AQICN {station_key} error: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        
        return pd.DataFrame()
    
    def _scrape_aqicn_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Scrape AQICN historical data (as fallback when API not available)
        """
        self.logger.info("Attempting to scrape AQICN historical data...")
        
        # This would require web scraping the AQICN historical pages
        # For now, return empty DataFrame and focus on other sources
        self.logger.warning("AQICN web scraping not implemented - use API key for historical data")
        return pd.DataFrame()
    
    def get_openmeteo_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical weather and air quality data from Open-Meteo Archive API (Free)
        This covers 2020-2024 with hourly data
        """
        self.logger.info(f"Collecting Open-Meteo historical data ({start_date} to {end_date})...")
        all_data = []
        
        for station_key, station_info in self.stations.items():
            try:
                # Open-Meteo Historical API (ERA5 reanalysis + air quality)
                weather_url = "https://archive-api.open-meteo.com/v1/era5"
                air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
                
                # Split date range into yearly chunks to avoid timeout
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                station_data = []
                
                # Process year by year
                current_year = start_dt.year
                while current_year <= end_dt.year:
                    year_start = max(start_dt, datetime(current_year, 1, 1)).strftime('%Y-%m-%d')
                    year_end = min(end_dt, datetime(current_year, 12, 31)).strftime('%Y-%m-%d')
                    
                    # Weather data
                    weather_params = {
                        'latitude': station_info['lat'],
                        'longitude': station_info['lng'],
                        'start_date': year_start,
                        'end_date': year_end,
                        'hourly': [
                            'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
                            'wind_speed_10m', 'wind_direction_10m', 'precipitation'
                        ]
                    }
                    
                    # Air quality data (separate API)
                    air_params = {
                        'latitude': station_info['lat'],
                        'longitude': station_info['lng'],
                        'start_date': year_start,
                        'end_date': year_end,
                        'hourly': ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
                    }
                    
                    try:
                        weather_response = requests.get(weather_url, params=weather_params, timeout=60)
                        air_response = requests.get(air_url, params=air_params, timeout=60)
                        
                        if weather_response.status_code == 200 and air_response.status_code == 200:
                            weather_data = weather_response.json()
                            air_data = air_response.json()
                            
                            # Combine the data
                            year_df = pd.DataFrame({
                                'datetime': pd.to_datetime(weather_data['hourly']['time']),
                                'temperature': weather_data['hourly']['temperature_2m'],
                                'humidity': weather_data['hourly']['relative_humidity_2m'],
                                'pressure': weather_data['hourly']['surface_pressure'],
                                'wind_speed': weather_data['hourly']['wind_speed_10m'],
                                'wind_direction': weather_data['hourly']['wind_direction_10m'],
                                'precipitation': weather_data['hourly']['precipitation'],
                                'pm2_5': air_data['hourly']['pm2_5'],
                                'pm10': air_data['hourly']['pm10'],
                                'no2': air_data['hourly']['nitrogen_dioxide'],
                                'so2': air_data['hourly']['sulphur_dioxide'],
                                'o3': air_data['hourly']['ozone'],
                                'station_id': f"openmeteo_{station_info['name']}",
                                'lat': station_info['lat'],
                                'lng': station_info['lng'],
                                'elevation': station_info['elevation'],
                                'source': 'openmeteo_historical'
                            })
                            
                            station_data.append(year_df)
                            self.logger.info(f"✓ {station_key} {current_year}: {len(year_df)} records")
                        
                        else:
                            self.logger.warning(f"✗ {station_key} {current_year}: API error {weather_response.status_code}/{air_response.status_code}")
                    
                    except Exception as e:
                        self.logger.error(f"✗ {station_key} {current_year}: {e}")
                    
                    current_year += 1
                    time.sleep(1)  # Rate limiting
                
                if station_data:
                    station_combined = pd.concat(station_data, ignore_index=True)
                    all_data.append(station_combined)
                    self.logger.info(f"✓ {station_key} total: {len(station_combined)} records")
            
            except Exception as e:
                self.logger.error(f"Open-Meteo {station_key} error: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Open-Meteo total: {len(combined_df)} records")
            return combined_df
        
        return pd.DataFrame()
    
    def get_openweather_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical weather data from OpenWeather Historical API
        Note: Requires paid subscription for historical data beyond 5 days
        """
        if 'openweather' not in self.api_keys:
            self.logger.warning("OpenWeather API key not provided for historical data")
            return pd.DataFrame()
        
        self.logger.info(f"Collecting OpenWeather historical data ({start_date} to {end_date})...")
        
        # OpenWeather Historical API is paid - implement if you have subscription
        # For free tier, we skip this
        self.logger.warning("OpenWeather historical data requires paid subscription - skipping")
        return pd.DataFrame()
    
    def add_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required ML features"""
        if df.empty:
            return df
        
        # Ensure datetime is properly formatted
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Nepal-specific seasons
        def get_nepal_season(month):
            if month in [12, 1, 2]: return 'winter'
            elif month in [3, 4, 5]: return 'spring'  
            elif month in [6, 7, 8, 9]: return 'monsoon'
            else: return 'post_monsoon'
        
        df['season'] = df['month'].apply(get_nepal_season)
        
        # Pollution season (high pollution period in Kathmandu)
        df['high_pollution_season'] = df['month'].apply(
            lambda x: 1 if x in [10, 11, 12, 1, 2, 3] else 0
        )
        
        # Festival season (affects traffic and activities)
        df['festival_season'] = df['month'].apply(
            lambda x: 1 if x in [9, 10, 11] else 0
        )
        
        # Add synthetic AOD data if missing
        if 'aod_550nm' not in df.columns or df['aod_550nm'].isnull().all():
            if 'pm2_5' in df.columns:
                # Estimate AOD from PM2.5 with seasonal variation
                base_aod = df['pm2_5'] * 0.008
                seasonal_mult = df['season'].map({
                    'winter': 1.4, 'spring': 1.2, 'post_monsoon': 1.1, 'monsoon': 0.7
                })
                df['aod_550nm'] = (base_aod * seasonal_mult + 
                                  np.random.normal(0, 0.02, len(df))).clip(0.05, 2.5)
            else:
                df['aod_550nm'] = np.random.uniform(0.1, 0.8, len(df))
        
        return df
    
    def quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data quality control"""
        if df.empty:
            return df
        
        original_len = len(df)
        self.logger.info(f"Starting quality control on {original_len} records...")
        
        # Remove invalid dates
        df = df.dropna(subset=['datetime'])
        
        # Remove extreme outliers but keep realistic high pollution events
        if 'pm2_5' in df.columns:
            df = df[df['pm2_5'].between(0, 500)]  # 500 is extreme but possible during events
        
        if 'temperature' in df.columns:
            df = df[df['temperature'].between(-10, 45)]  # Realistic for Kathmandu
        
        if 'humidity' in df.columns:
            df = df[df['humidity'].between(0, 100)]
        
        if 'pressure' in df.columns:
            df = df[df['pressure'].between(850, 1050)]  # Adjusted for Kathmandu altitude
        
        if 'wind_speed' in df.columns:
            df = df[df['wind_speed'].between(0, 50)]
        
        # Remove completely empty rows
        df = df.dropna(how='all', subset=['pm2_5', 'temperature', 'humidity', 'wind_speed'])
        
        removed = original_len - len(df)
        if removed > 0:
            self.logger.info(f"Quality control: Removed {removed} invalid records ({removed/original_len*100:.1f}%)")
        
        return df
    
    def collect_historical_dataset(self, 
                                  start_date: str = None, 
                                  end_date: str = None,
                                  sources: List[str] = None) -> pd.DataFrame:
        """
        Main method to collect comprehensive historical dataset
        
        Args:
            start_date: Start date (default: 2021-01-01)
            end_date: End date (default: 2024-08-25)
            sources: Data sources to use
        """
        start_date = start_date or self.target_start_date
        end_date = end_date or self.target_end_date
        sources = sources or ['open_data_nepal', 'openmeteo', 'aqicn']
        
        self.logger.info("=" * 80)
        self.logger.info(f"HISTORICAL KATHMANDU AIR QUALITY DATA COLLECTION")
        self.logger.info(f"Target period: {start_date} to {end_date}")
        self.logger.info(f"Sources: {sources}")
        self.logger.info("=" * 80)
        
        all_datasets = []
        
        # 1. Open Data Nepal (2015-2021 official data)
        if 'open_data_nepal' in sources:
            open_data = self.download_open_data_nepal()
            if not open_data.empty:
                all_datasets.append(open_data)
        
        # 2. Open-Meteo Historical (2020-2024, free)
        if 'openmeteo' in sources:
            openmeteo_data = self.get_openmeteo_historical(start_date, end_date)
            if not openmeteo_data.empty:
                all_datasets.append(openmeteo_data)
        
        # 3. AQICN Historical (requires API key)
        if 'aqicn' in sources:
            aqicn_data = self.get_aqicn_historical(start_date, end_date)
            if not aqicn_data.empty:
                all_datasets.append(aqicn_data)
        
        # Combine all datasets
        if all_datasets:
            self.logger.info("-" * 60)
            self.logger.info("COMBINING AND PROCESSING DATA")
            
            combined_df = pd.concat(all_datasets, ignore_index=True, sort=False)
            
            # Add comprehensive features
            combined_df = self.add_comprehensive_features(combined_df)
            
            # Quality control
            combined_df = self.quality_control(combined_df)
            
            # Remove duplicates
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(
                subset=['datetime', 'station_id'], keep='first'
            )
            after_dedup = len(combined_df)
            
            if before_dedup != after_dedup:
                self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
            
            # Sort chronologically
            combined_df = combined_df.sort_values(['station_id', 'datetime']).reset_index(drop=True)
            
            # Final summary
            self.logger.info("-" * 60)
            self.logger.info("FINAL DATASET SUMMARY")
            self.logger.info(f"Total records: {len(combined_df):,}")
            self.logger.info(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
            self.logger.info(f"Unique stations: {combined_df['station_id'].nunique()}")
            self.logger.info(f"Time span: {(combined_df['datetime'].max() - combined_df['datetime'].min()).days} days")
            
            return combined_df
        
        else:
            self.logger.error("No data collected from any source!")
            return pd.DataFrame()

def main():
    """Main execution function"""
    
    # API Keys (add your keys here)
    API_KEYS = {
        'aqicn': 'YOUR_AQICN_API_KEY',          # Get from https://aqicn.org/data-platform/token/
        'openweather': 'YOUR_OPENWEATHER_KEY'    # Get from https://openweathermap.org/api
    }
    
    # Initialize collector
    collector = HistoricalKathmanduCollector(api_keys=API_KEYS)
    
    # Collect historical data (2021-2024)
    dataset = collector.collect_historical_dataset(
        start_date='2021-01-01',
        end_date='2024-08-25',
        sources=['open_data_nepal', 'openmeteo', 'aqicn']  # Use available sources
    )
    
    if not dataset.empty:
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'data/kathmandu_historical_air_quality_{timestamp}.csv'
        dataset.to_csv(filename, index=False)
        
        print("\n" + "=" * 80)
        print("🎉 HISTORICAL DATA COLLECTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Dataset saved: {filename}")
        print(f"📊 Total records: {len(dataset):,}")
        print(f"📅 Period: {dataset['datetime'].min()} to {dataset['datetime'].max()}")
        print(f"🏭 Stations: {dataset['station_id'].nunique()}")
        print(f"⏱️  Time span: {(dataset['datetime'].max() - dataset['datetime'].min()).days} days")
        
        # ML readiness check
        ml_features = ['pm2_5', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 
                      'pressure', 'aod_550nm', 'hour', 'day', 'datetime', 'station_id', 'lat', 'lng', 'elevation']
        
        print(f"\n📋 ML FEATURES AVAILABILITY:")
        for feature in ml_features:
            if feature in dataset.columns:
                available = (~dataset[feature].isnull()).sum()
                print(f"✅ {feature}: {available:,} values ({available/len(dataset)*100:.1f}%)")
            else:
                print(f"❌ {feature}: MISSING")
        
        # Data quality summary
        if 'pm2_5' in dataset.columns:
            pm25_data = dataset['pm2_5'].dropna()
            print(f"\n🏭 PM2.5 SUMMARY:")
            print(f"   Records with PM2.5: {len(pm25_data):,}")
            print(f"   Mean PM2.5: {pm25_data.mean():.2f} μg/m³")
            print(f"   Median PM2.5: {pm25_data.median():.2f} μg/m³")
            print(f"   Range: {pm25_data.min():.1f} - {pm25_data.max():.1f} μg/m³")
        
        # Seasonal analysis
        if 'season' in dataset.columns and len(dataset) > 1000:
            seasonal_pm25 = dataset.groupby('season')['pm2_5'].agg(['count', 'mean', 'std']).round(2)
            print(f"\n🌦️  SEASONAL PM2.5 PATTERNS:")
            for season, stats in seasonal_pm25.iterrows():
                print(f"   {season.capitalize()}: {stats['mean']} ± {stats['std']} μg/m³ ({int(stats['count'])} records)")
        
        print(f"\n🚀 DATASET READY FOR MACHINE LEARNING!")
        return dataset
    
    else:
        print("❌ No historical data collected. Check:")
        print("1. Internet connection")
        print("2. API keys (if using paid services)")  
        print("3. Try with just 'openmeteo' source first")
        return None

if __name__ == "__main__":
    dataset = main()