from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "PlayTennis.csv"

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CSV_PATH}. Keep PlayTennis.csv beside the Python files."
        )
    return pd.read_csv(CSV_PATH)

def encode_data(data):
    encoded = data.copy()
    encoders = {}
    for column in encoded.columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder
    return encoded, encoders

def train_model(encoded):
    y = encoded["Play Tennis"]
    X = encoded.drop(columns=["Play Tennis"])
    model = DecisionTreeClassifier(criterion="entropy", random_state=0)
    model.fit(X, y)
    return model, X, y
