import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    def __init__(self, data_dir='data/raw', processed_dir='data/processed', chunksize=50000):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.chunksize = chunksize
        
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_nyc_taxi_data(self, file_path):
        """Load NYC taxi data in chunks with logging"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data from {file_path}...")
        
        chunks = []
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunksize,
                                             parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])):
            chunk = chunk[(chunk['trip_distance'] > 0) & (chunk['fare_amount'] > 0)]
            chunk = chunk[(chunk['pickup_latitude'] != 0) & (chunk['pickup_longitude'] != 0)]
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing chunk {i + 1}...")
        
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data loaded. Total rows after cleaning: {len(df)}")
        return df

    def create_zones(self, df, grid_size=10):
        """Create geographic zones using lat/lon"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating zones with grid size {grid_size}...")
        
        lat_min, lat_max = df['pickup_latitude'].min(), df['pickup_latitude'].max()
        lon_min, lon_max = df['pickup_longitude'].min(), df['pickup_longitude'].max()
        
        lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
        lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)
        
        df['pickup_zone'] = pd.cut(df['pickup_latitude'], bins=lat_bins, labels=False, duplicates='drop')
        df['dropoff_zone'] = pd.cut(df['dropoff_longitude'], bins=lon_bins, labels=False, duplicates='drop')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Zones created.")
        return df, lat_bins, lon_bins

    def aggregate_demand(self, df, time_interval_minutes=10):
        """Aggregate demand by time and zones"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregating demand in {time_interval_minutes}-minute intervals...")
        
        df['time_slot'] = df['tpep_pickup_datetime'].dt.floor(f'{time_interval_minutes}min')
        demand = df.groupby(['time_slot', 'pickup_zone', 'dropoff_zone']).size().reset_index(name='demand')
        
        # Save aggregated demand
        demand_file = os.path.join(self.processed_dir, 'demand.csv')
        demand.to_csv(demand_file, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Demand aggregation done. Total records: {len(demand)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved demand to {demand_file}")
        return demand

    def extract_features(self, df):
        """Extract features for ML model"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting features...")
        features = []
        
        for i, row in enumerate(df.itertuples(index=False), 1):
            hour = row.tpep_pickup_datetime.hour
            day_of_week = row.tpep_pickup_datetime.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            feature_dict = {
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'trip_distance': row.trip_distance,
                'fare_amount': row.fare_amount,
                'passenger_count': row.passenger_count,
                'pickup_zone': row.pickup_zone,
                'dropoff_zone': row.dropoff_zone
            }
            features.append(feature_dict)
            
            if i % 50000 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i} rows for features...")
        
        features_df = pd.DataFrame(features)
        
        # Save features
        features_file = os.path.join(self.processed_dir, 'features.csv')
        features_df.to_csv(features_file, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature extraction complete. Total features: {len(features_df)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved features to {features_file}")
        
        return features_df

# Usage
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_nyc_taxi_data('data/raw/yellow_tripdata_2013-03.csv')
    df, lat_bins, lon_bins = loader.create_zones(df, grid_size=10)
    demand = loader.aggregate_demand(df, time_interval_minutes=10)
    features = loader.extract_features(df)
