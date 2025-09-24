import os
import pandas as pd
import sqlite3
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import numpy as np

DB_PATH = "expenses.db"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM expenses", conn, parse_dates=["date"])
    conn.close()
    return df

def train_category_model(df):
    df_text = df.dropna(subset=["description", "category"])
    if df_text.shape[0] < 10:
        print("Not enough labeled description data for a good classifier. Need >=10 rows.")
    X = df_text["description"].astype(str)
    y = df_text["category"].astype(str)
    pipe = make_pipeline(CountVectorizer(), MultinomialNB())
    pipe.fit(X, y)
    # quick eval (train set)
    preds = pipe.predict(X)
    print("Category classifier — train report:")
    print(classification_report(y, preds))
    joblib.dump(pipe, os.path.join(MODELS_DIR, "cat_model.pkl"))
    print("Saved category model -> models/cat_model.pkl")

def train_prediction_model(df):
    if df.empty:
        print("No expense data for prediction model.")
        return
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().reset_index()
    monthly = monthly.sort_values('date').reset_index(drop=True)
    if len(monthly) < 3:
        print("Not enough monthly points to train prediction model (need >=3).")
        # still save a fallback model that predicts mean
        mean_val = monthly['amount'].mean() if not monthly.empty else 0.0
        joblib.dump(("mean", mean_val), os.path.join(MODELS_DIR, "pred_model.pkl"))
        print("Saved fallback pred_model (mean).")
        return

    monthly['idx'] = np.arange(len(monthly))
    X = monthly[['idx']].values  # simple time index
    y = monthly['amount'].values
    lr = LinearRegression()
    lr.fit(X, y)
    # Save also the latest length so app can compute next idx
    meta = {"model": lr, "last_idx": int(monthly['idx'].iloc[-1]), "monthly_dates": monthly['date'].dt.strftime("%Y-%m").tolist(), "monthly_values": monthly['amount'].tolist()}
    joblib.dump(meta, os.path.join(MODELS_DIR, "pred_model.pkl"))
    print("Saved prediction model -> models/pred_model.pkl")

def train_anomaly_model(df):
    if df.empty:
        print("No data for anomaly model.")
        return
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().reset_index()
    if len(monthly) < 3:
        print("Not enough monthly points for anomaly detection.")
        return
    X = monthly['amount'].values.reshape(-1,1)
    iso = IsolationForest(random_state=42, contamination=0.1)
    iso.fit(X)
    joblib.dump({"model": iso, "dates": monthly['date'].dt.strftime("%Y-%m").tolist(), "values": monthly['amount'].tolist()}, os.path.join(MODELS_DIR, "iso_model.pkl"))
    print("Saved anomaly model -> models/iso_model.pkl")

def train_all():
    df = load_data()
    print(f"Loaded {len(df)} rows from DB.")
    train_category_model(df)
    train_prediction_model(df)
    train_anomaly_model(df)
    print("Training completed.")

if __name__ == "__main__":
    train_all()
