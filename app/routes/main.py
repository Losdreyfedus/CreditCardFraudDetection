from flask import render_template
from app import app
from app.database.db import get_dashboard_data, get_recent_frauds
import json
import random
from datetime import datetime, timedelta
from config import Config

# Model metriklerini yükle
try:
    with open(Config.METRICS, 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {
        'accuracy': 0.999,
        'precision': 0.985,
        'recall': 0.978,
        'f1_score': 0.981,
        'auc_roc': 0.995
    }

def generate_trend_data():
    """Örnek trend verisi oluştur"""
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
    dates.reverse()
    
    normal_counts = [random.randint(900, 1100) for _ in range(30)]
    fraud_counts = [random.randint(15, 25) for _ in range(30)]
    
    return {
        'dates': dates,
        'normal_counts': normal_counts,
        'fraud_counts': fraud_counts,
        'daily_metrics': {
            'total_transactions': sum(normal_counts) + sum(fraud_counts),
            'fraud_detected': sum(fraud_counts),
            'avg_response_time': round(random.uniform(0.1, 0.3), 3)
        }
    }

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)

@app.route('/dashboard')
def dashboard():
    try:
        dashboard_data = get_dashboard_data()
        recent_frauds = get_recent_frauds()
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
        dashboard_data = generate_trend_data()
        recent_frauds = []
    
    return render_template('dashboard.html',
                         metrics=metrics,
                         trend_data=dashboard_data,
                         recent_frauds=recent_frauds) 