import os

class Config:
    # Flask ayarları
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'gizli-anahtar-buraya'
    
    # Veritabanı ayarları
    DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fraud_detection.db')
    
    # Model dosyaları
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    BEST_MODEL = os.path.join(MODEL_PATH, 'best_model.joblib')
    SCALER = os.path.join(MODEL_PATH, 'scaler.joblib')
    METRICS = os.path.join(MODEL_PATH, 'metrics.json')
    
    # Static dosyalar
    STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static') 