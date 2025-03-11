import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from contextlib import contextmanager
from config import Config

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Veritabanını oluştur"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # İşlemler tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            is_fraud BOOLEAN NOT NULL,
            merchant_category TEXT,
            location TEXT,
            risk_score INTEGER
        )
        ''')
        
        # Performans metrikleri tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date DATE PRIMARY KEY,
            total_transactions INTEGER,
            fraud_detected INTEGER,
            avg_response_time REAL
        )
        ''')
        
        conn.commit()

def get_dashboard_data():
    """Veritabanından dashboard verilerini çek"""
    with get_db_connection() as conn:
        # Son 30 günün verilerini çek
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Günlük işlem sayıları
        daily_counts = pd.read_sql_query('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_count,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
            FROM transactions 
            WHERE DATE(timestamp) >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', conn, params=[thirty_days_ago])
        
        # Performans metrikleri
        metrics = pd.read_sql_query('''
            SELECT * FROM daily_metrics
            WHERE date >= ?
            ORDER BY date DESC
            LIMIT 1
        ''', conn, params=[thirty_days_ago]).to_dict('records')[0]
        
        return {
            'dates': daily_counts['date'].tolist(),
            'normal_counts': (daily_counts['total_count'] - daily_counts['fraud_count']).tolist(),
            'fraud_counts': daily_counts['fraud_count'].tolist(),
            'daily_metrics': metrics
        }

def get_recent_frauds():
    """Son tespit edilen dolandırıcılık vakalarını getir"""
    with get_db_connection() as conn:
        recent_frauds = pd.read_sql_query('''
            SELECT 
                timestamp,
                amount,
                risk_score,
                merchant_category,
                location
            FROM transactions 
            WHERE is_fraud = 1
            ORDER BY timestamp DESC
            LIMIT 5
        ''', conn)
        
        # Timestamp'i datetime nesnesine dönüştür ve formatla
        recent_frauds['timestamp'] = pd.to_datetime(recent_frauds['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return recent_frauds.to_dict('records')