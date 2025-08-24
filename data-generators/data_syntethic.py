import json
import random
from datetime import datetime, timedelta
import math
import psycopg2
from psycopg2.extras import DictCursor
import os
from typing import Optional, Dict, List

class DatabaseWeatherAQIGenerator:
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize the generator with database configuration
        
        Args:
            db_config: Dictionary with database connection parameters
                      If None, will try to read from environment variables
        """
        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'your_database'),
                'user': os.getenv('DB_USER', 'your_user'),
                'password': os.getenv('DB_PASSWORD', 'your_password'),
                'port': os.getenv('DB_PORT', '5432')
            }
        else:
            self.db_config = db_config
            
        # Base patterns for different hours (0-23)
        self.aqi_pattern = {
            # Midnight to early morning (lower traffic, cooler temps)
            0: 0.6, 1: 0.5, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.6,
            # Morning rush hour (increased traffic)
            6: 0.8, 7: 1.0, 8: 1.1, 9: 1.0, 10: 0.9,
            # Midday (peak temperature, some industrial activity)
            11: 0.9, 12: 1.0, 13: 1.1, 14: 1.0, 15: 0.9,
            # Afternoon
            16: 0.9, 17: 1.0, 18: 1.1, 19: 1.0, 20: 0.9,
            # Evening (decreasing activity)
            21: 0.8, 22: 0.7, 23: 0.6
        }
        
        self.temp_pattern = {
            # Night time temperatures (cooler)
            0: 0.7, 1: 0.65, 2: 0.6, 3: 0.55, 4: 0.6, 5: 0.65,
            # Morning warming
            6: 0.7, 7: 0.8, 8: 0.85, 9: 0.9, 10: 0.95,
            # Peak daytime temperatures
            11: 1.0, 12: 1.05, 13: 1.1, 14: 1.1, 15: 1.05,
            # Afternoon cooling
            16: 1.0, 17: 0.95, 18: 0.9, 19: 0.85, 20: 0.8,
            # Evening cooling
            21: 0.75, 22: 0.7, 23: 0.65
        }
        
        self.humidity_pattern = {
            # Higher humidity at night and early morning
            0: 1.2, 1: 1.25, 2: 1.3, 3: 1.35, 4: 1.3, 5: 1.25,
            # Decreasing as temperature rises
            6: 1.1, 7: 1.0, 8: 0.95, 9: 0.9, 10: 0.85,
            # Lowest during peak heat
            11: 0.8, 12: 0.75, 13: 0.7, 14: 0.75, 15: 0.8,
            # Gradually increasing
            16: 0.85, 17: 0.9, 18: 0.95, 19: 1.0, 20: 1.05,
            # Back to high at night
            21: 1.1, 22: 1.15, 23: 1.2
        }

    def get_db_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            raise Exception(f"Failed to connect to database: {e}")

    def fetch_latest_observation(self, city: str) -> Optional[Dict]:
        """
        Fetch the most recent observation for a given city
        
        Args:
            city: City name to search for
            
        Returns:
            Dictionary with the latest observation data or None if not found
        """
        query = """
        SELECT 
            city,
            pm25,
            temperature,
            wind,
            humidity,
            aqi,
            observed_at,
            pressure,
            dew,
            rainfall
        FROM aqi_observations 
        WHERE city ILIKE %s 
        ORDER BY observed_at DESC 
        LIMIT 1
        """
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(query, (f"%{city}%",))
                    row = cur.fetchone()
                    
                    if row:
                        return dict(row)
                    return None
                    
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return None

    def fetch_recent_observations(self, city: str, hours: int = 24) -> List[Dict]:
        """
        Fetch recent observations for a city to calculate averages
        
        Args:
            city: City name to search for
            hours: Number of hours back to look
            
        Returns:
            List of observation dictionaries
        """
        query = """
        SELECT 
            city,
            pm25,
            temperature,
            wind,
            humidity,
            aqi,
            observed_at,
            pressure,
            dew,
            rainfall
        FROM aqi_observations 
        WHERE city ILIKE %s 
        AND observed_at >= %s
        ORDER BY observed_at DESC
        """
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(query, (f"%{city}%", cutoff_time))
                    rows = cur.fetchall()
                    
                    return [dict(row) for row in rows]
                    
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return []

    def get_base_data_from_db(self, city: str) -> Optional[Dict]:
        """
        Get base data from database for synthetic generation
        
        Args:
            city: City name to fetch data for
            
        Returns:
            Base data dictionary in the expected format or None if not found
        """
        latest = self.fetch_latest_observation(city)
        
        if not latest:
            print(f"No data found for city: {city}")
            return None
        
        recent = self.fetch_recent_observations(city, hours=24)
        
        if recent:
            avg_pm25 = sum(r['pm25'] for r in recent if r['pm25'] is not None) / len([r for r in recent if r['pm25'] is not None])
            avg_temp = sum(r['temperature'] for r in recent if r['temperature'] is not None) / len([r for r in recent if r['temperature'] is not None])
            avg_wind = sum(r['wind'] for r in recent if r['wind'] is not None) / len([r for r in recent if r['wind'] is not None])
            avg_humidity = sum(r['humidity'] for r in recent if r['humidity'] is not None) / len([r for r in recent if r['humidity'] is not None])
        else:
            avg_pm25 = latest['pm25'] or 50
            avg_temp = latest['temperature'] or 25
            avg_wind = latest['wind'] or 5
            avg_humidity = latest['humidity'] or 60
        
        base_data = {
            "status": "success",
            "city": latest['city'],
            "data": {
                "pm25": {
                    "value": int(avg_pm25) if avg_pm25 else 50,
                    "observed_at": latest['observed_at'].isoformat() if latest['observed_at'] else datetime.utcnow().isoformat()
                },
                "temp": {
                    "value": round(float(avg_temp), 1) if avg_temp else 25.0,
                    "observed_at": latest['observed_at'].isoformat() if latest['observed_at'] else datetime.utcnow().isoformat()
                },
                "wind": {
                    "value": round(float(avg_wind), 1) if avg_wind else 5.0,
                    "observed_at": latest['observed_at'].isoformat() if latest['observed_at'] else datetime.utcnow().isoformat()
                },
                "humidity": {
                    "value": round(float(avg_humidity), 1) if avg_humidity else 60.0,
                    "observed_at": latest['observed_at'].isoformat() if latest['observed_at'] else datetime.utcnow().isoformat()
                }
            },
            "latest_observed_at": latest['observed_at'].isoformat() if latest['observed_at'] else datetime.utcnow().isoformat()
        }
        
        return base_data

    def list_available_cities(self, limit: int = 20) -> List[str]:
        """
        Get a list of cities available in the database
        
        Args:
            limit: Maximum number of cities to return
            
        Returns:
            List of city names
        """
        query = """
        SELECT DISTINCT city 
        FROM aqi_observations 
        ORDER BY MAX(observed_at) DESC
        LIMIT %s
        """
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (limit,))
                    cities = [row[0] for row in cur.fetchall()]
                    return cities
                    
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return []

    def generate_hourly_data(self, base_data, start_date, hours=24):
        """
        Generate hourly synthetic data based on base values
        
        Args:
            base_data: Dictionary containing base values for pm25, temp, wind, humidity
            start_date: Starting datetime (string in ISO format)
            hours: Number of hours to generate (default 24)
        
        Returns:
            List of hourly data points
        """
        hourly_data = []
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        
        base_pm25 = base_data['data']['pm25']['value']
        base_temp = base_data['data']['temp']['value']
        base_wind = base_data['data']['wind']['value']
        base_humidity = base_data['data']['humidity']['value']
        city = base_data['city']
        
        for hour in range(hours):
            current_time = start_dt + timedelta(hours=hour)
            hour_of_day = current_time.hour
            
            pm25_multiplier = self.aqi_pattern[hour_of_day]
            temp_multiplier = self.temp_pattern[hour_of_day]
            humidity_multiplier = self.humidity_pattern[hour_of_day]
            
            pm25_variation = random.uniform(0.85, 1.15)
            temp_variation = random.uniform(0.95, 1.05)
            wind_variation = random.uniform(0.7, 1.3)
            humidity_variation = random.uniform(0.9, 1.1)
            
            pm25_value = round(base_pm25 * pm25_multiplier * pm25_variation)
            temp_value = round(base_temp * temp_multiplier * temp_variation, 1)
            wind_value = round(base_wind * wind_variation, 1)
            humidity_value = round(base_humidity * humidity_multiplier * humidity_variation, 1)
            
            pm25_value = max(5, min(300, pm25_value))  
            temp_value = max(15, min(45, temp_value))   
            wind_value = max(0, min(15, wind_value))    
            humidity_value = max(20, min(100, humidity_value))  
            
            hourly_entry = {
                "status": "success",
                "city": city,
                "data": {
                    "pm25": {
                        "value": pm25_value,
                        "observed_at": current_time.strftime("%Y-%m-%dT%H:00:00+00:00")
                    },
                    "temp": {
                        "value": temp_value,
                        "observed_at": current_time.strftime("%Y-%m-%dT%H:00:00+00:00")
                    },
                    "wind": {
                        "value": wind_value,
                        "observed_at": current_time.strftime("%Y-%m-%dT%H:00:00+00:00")
                    },
                    "humidity": {
                        "value": humidity_value,
                        "observed_at": current_time.strftime("%Y-%m-%dT%H:00:00+00:00")
                    }
                },
                "latest_observed_at": current_time.strftime("%Y-%m-%dT%H:00:00+00:00")
            }
            
            hourly_data.append(hourly_entry)
        
        return hourly_data

    def save_to_json(self, data, filename):
        """Save the generated data to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'your_database_name',
        'user': 'your_username',
        'password': 'your_password',
        'port': '5432'
    }
    
    generator = DatabaseWeatherAQIGenerator(db_config)
    
    print("Available cities in database:")
    cities = generator.list_available_cities()
    for i, city in enumerate(cities[:10], 1):
        print(f"{i}. {city}")
    
    if not cities:
        print("No cities found in database. Please check your database connection and data.")
        exit(1)
    
    target_city = cities[0]  
    print(f"\nUsing city: {target_city}")

    base_data = generator.get_base_data_from_db(target_city)
    
    if not base_data:
        print(f"Could not fetch data for {target_city}")
        exit(1)
    
    print("\nBase data from database:")
    print(json.dumps(base_data, indent=2))
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    
    hourly_data = generator.generate_hourly_data(base_data, start_date, hours=24)

    print("\nSample generated data:")
    for i in range(3):
        print(f"\nHour {i}:")
        print(json.dumps(hourly_data[i], indent=2))
    
    filename = f"synthetic_hourly_{target_city.replace(',', '').replace(' ', '_').lower()}.json"
    generator.save_to_json(hourly_data, filename)
    
    print("\n" + "="*50)
    print("Generating 7 days of data...")
    
    all_week_data = []
    for day in range(7):
        day_start = today + timedelta(days=day)
        day_data = generator.generate_hourly_data(
            base_data, 
            day_start.strftime("%Y-%m-%dT00:00:00+00:00"), 
            hours=24
        )
        all_week_data.extend(day_data)
    
    weekly_filename = f"synthetic_weekly_{target_city.replace(',', '').replace(' ', '_').lower()}.json"
    generator.save_to_json(all_week_data, weekly_filename)
    print(f"Generated {len(all_week_data)} hours of data (7 days)")