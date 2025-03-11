import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np

def clear_database():
    """Veritabanındaki tüm verileri temizle"""
    with sqlite3.connect('fraud_detection.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM transactions')
        cursor.execute('DELETE FROM daily_metrics')
        conn.commit()

def insert_sample_data():
    """Örnek veriler ekle"""
    with sqlite3.connect('fraud_detection.db') as conn:
        cursor = conn.cursor()
        
        # Tabloyu yeniden oluştur
        cursor.execute('DROP TABLE IF EXISTS transactions')
        cursor.execute('''
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            is_fraud BOOLEAN NOT NULL,
            merchant_category TEXT,
            location TEXT,
            risk_score INTEGER
        )
        ''')
        
        # Son 30 gün için veriler oluştur
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # İşlem verileri
        for day in range(31):
            current_date = start_date + timedelta(days=day)
            
            # Normal işlemler (900-1100 arası)
            normal_count = random.randint(900, 1100)
            for _ in range(normal_count):
                timestamp = current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                amount = random.uniform(10, 1000)
                merchant = random.choice(['Market', 'Restaurant', 'Online Store', 'Gas Station', 'Electronics'])
                location = random.choice(['Istanbul', 'Ankara', 'Izmir', 'Antalya', 'Bursa'])
                risk_score = random.randint(5, 30)  # Düşük risk skoru
                
                cursor.execute('''
                INSERT INTO transactions (timestamp, amount, is_fraud, merchant_category, location, risk_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, amount, 0, merchant, location, risk_score))
            
            # Dolandırıcılık işlemleri (15-25 arası)
            fraud_count = random.randint(15, 25)
            for _ in range(fraud_count):
                timestamp = current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                amount = random.uniform(500, 5000)  # Daha yüksek tutarlar
                merchant = random.choice(['Online Store', 'ATM', 'Foreign Merchant'])
                location = random.choice(['Foreign', 'Unknown', 'Multiple Locations'])
                risk_score = random.randint(75, 95)  # Yüksek risk skoru
                
                cursor.execute('''
                INSERT INTO transactions (timestamp, amount, is_fraud, merchant_category, location, risk_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, amount, 1, merchant, location, risk_score))
            
            # Günlük metrikler
            avg_response_time = random.uniform(0.01, 0.1)
            cursor.execute('''
            INSERT INTO daily_metrics (date, total_transactions, fraud_detected, avg_response_time)
            VALUES (?, ?, ?, ?)
            ''', (current_date.date(), normal_count + fraud_count, fraud_count, avg_response_time))
        
        conn.commit()
        print("Örnek veriler başarıyla eklendi!")

if __name__ == '__main__':
    clear_database()
    insert_sample_data() 