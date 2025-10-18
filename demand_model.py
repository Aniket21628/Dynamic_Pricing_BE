import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

class DemandModel:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()

    def prepare_features(self, features_df):
        X = features_df[['hour', 'day_of_week', 'is_weekend',
                         'passenger_count', 'pickup_zone', 'dropoff_zone']]
        y = features_df['fare_amount']
        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"[INFO] Training model on {len(X_train)} rows (may take a few minutes)...")
        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"[INFO] Train R² Score: {train_score:.4f}")
        print(f"[INFO] Test R² Score: {test_score:.4f}")
        return train_score, test_score

    def predict_demand(self, features):
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def estimate_elasticity(self, base_price, multipliers=[0.8, 1.0, 1.2, 1.5]):
        elasticity = {}
        for mult in multipliers:
            elasticity[mult] = self.acceptance_probability(mult)
        return elasticity

    def acceptance_probability(self, price_multiplier, demand_factor=1.0):
        base = 0.8 - 0.3 * (price_multiplier - 1)
        return max(0.05, min(1.0, base + 0.1 * demand_factor))

# ---------------- USAGE ----------------
if __name__ == "__main__":
    from data_loader import DataLoader

    features_csv_path = 'data/processed/features.csv'

    if os.path.exists(features_csv_path):
        print(f"[INFO] Loading features from {features_csv_path}...")
        features = pd.read_csv(features_csv_path)
    else:
        print(f"[INFO] Features CSV not found, extracting from raw data...")
        loader = DataLoader()
        df = loader.load_nyc_taxi_data('data/raw/yellow_tripdata_2013-03.csv')
        df, _, _ = loader.create_zones(df, grid_size=10)
        features = loader.extract_features(df)
        os.makedirs('data/processed', exist_ok=True)
        features.to_csv(features_csv_path, index=False)
        print(f"[INFO] Saved features to {features_csv_path}")

    demand_model = DemandModel()
    X, y = demand_model.prepare_features(features)
    demand_model.train(X, y)

    print("\n[INFO] Elasticity Analysis:")
    elasticity = demand_model.estimate_elasticity(base_price=1.0)
    for price_mult, acceptance in elasticity.items():
        print(f"Price Multiplier: {price_mult:.2f}, Acceptance: {acceptance:.4f}")
	