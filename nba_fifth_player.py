# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Step 1: Load the Datasets
file_paths = [
    "data/matchups-2007.csv", "data/matchups-2008.csv", "data/matchups-2009.csv",
    "data/matchups-2010.csv", "data/matchups-2011.csv", "data/matchups-2012.csv",
    "data/matchups-2013.csv", "data/matchups-2014.csv", "data/matchups-2015.csv"
]

# Read and merge all files
dfs = [pd.read_csv(file) for file in tqdm(file_paths, desc="Loading datasets")]
matchups_df = pd.concat(dfs, ignore_index=True)

# Step 2: Select Allowed Features
allowed_columns = [
    "game", "season", "home_team", "away_team",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4",
    "fga_home", "fta_home", "fgm_home", "fga_2_home", "fgm_2_home",
    "ast_home", "reb_home", "to_home", "pts_home", "pct_home",
    "outcome"
]

matchups_df = matchups_df[allowed_columns]

# Step 3: Handle Missing Values
matchups_df.fillna(0, inplace=True)

# Step 4: Convert Categorical Data to Numerical (Label Encoding)
label_encoders = {}
categorical_columns = ["home_team", "away_team"] + [f"home_{i}" for i in range(5)] + [f"away_{i}" for i in range(5)]

for col in tqdm(categorical_columns, desc="Encoding categorical variables"):
    le = LabelEncoder()
    matchups_df[col] = le.fit_transform(matchups_df[col])
    label_encoders[col] = le  # Save encoders for later decoding

# Step 5: Split Data for Training & Testing
features = [col for col in matchups_df.columns if col not in ["home_4", "game", "season"]]
target = "home_4"

X_train, X_test, y_train, y_test = train_test_split(matchups_df[features], matchups_df[target], test_size=0.2, random_state=42)

# Step 6: Train a Machine Learning Model with Progress Bar
class ProgressBarRandomForest(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        with tqdm(total=self.n_estimators, desc="Training RandomForest", unit="tree") as pbar:
            for i in range(self.n_estimators):
                super().fit(X, y, sample_weight=sample_weight)
                pbar.update(1)
        return self

# Reduce n_estimators to 50 for better performance
model = ProgressBarRandomForest(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 8: Predict the Fifth Player for Home Team
predictions = model.predict(X_test)

# Create Output DataFrame
output_df = X_test.copy()
output_df["predicted_home_4"] = predictions

# Step 9: Save Predictions to CSV
output_df.to_csv("nba_fifth_player_predictions.csv", index=False)

print("Predictions saved to 'nba_fifth_player_predictions.csv'")
