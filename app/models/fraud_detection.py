import joblib
from config import Config

# Model ve scaler'ı yükle
model = joblib.load(Config.BEST_MODEL)
scaler = joblib.load(Config.SCALER)

def preprocess_transaction(transaction_data):
    """
    Ham işlem verilerini model için hazırla
    
    Args:
        transaction_data (dict): Ham işlem verileri
        
    Returns:
        list: İşlenmiş özellikler
    """
    required_fields = [
        'amount', 'time', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
        'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
        'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28'
    ]
    
    # Gerekli alanların varlığını kontrol et
    for field in required_fields:
        if field not in transaction_data:
            raise ValueError(f"Eksik alan: {field}")
    
    # Özellikleri doğru sırayla al
    features = [float(transaction_data[field]) for field in required_fields]
    
    return features

def predict_transaction(transaction_data):
    """
    İşlem verilerini kullanarak dolandırıcılık tahmini yap
    
    Args:
        transaction_data (dict): İşlem verileri
        
    Returns:
        dict: Tahmin sonuçları
    """
    # Veriyi ön işle
    features = preprocess_transaction(transaction_data)
    
    # Tahmin yap
    scaled_features = scaler.transform([features])
    prediction = model.predict_proba(scaled_features)[0]
    
    return {
        'is_fraud': bool(prediction[1] > 0.5),
        'fraud_probability': float(prediction[1]),
        'risk_score': int(prediction[1] * 100)
    } 