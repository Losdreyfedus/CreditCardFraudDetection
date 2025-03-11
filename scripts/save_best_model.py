# En İyi Modeli Kaydet
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import os
import json
import matplotlib.pyplot as plt
from config import Config

def train_and_save_model():
    """En iyi modeli eğit ve kaydet"""
    print("Veri seti yükleniyor...")
    df = pd.read_csv('data/creditcard.csv')
    
    # Dolandırıcılık vakalarını al
    fraud_df = df[df['Class'] == 1]
    # Normal işlemlerden örnekleme yap
    normal_df = df[df['Class'] == 0].sample(n=len(fraud_df) * 2, random_state=42)
    
    # Verileri birleştir
    df_sample = pd.concat([fraud_df, normal_df])
    
    # Özellikler ve hedef değişkeni ayır
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE uygula
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # En iyi parametrelerle Gradient Boosting modelini eğit
    best_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    
    print("Model eğitiliyor...")
    best_model.fit(X_train_balanced, y_train_balanced)
    
    # Test seti üzerinde tahminler yap
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrikleri hesapla
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # ROC eğrisini çiz
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Gerekli dizinleri oluştur
    os.makedirs(os.path.dirname(Config.BEST_MODEL), exist_ok=True)
    os.makedirs(Config.STATIC_PATH, exist_ok=True)
    os.makedirs(os.path.join(Config.STATIC_PATH, 'images'), exist_ok=True)
    
    # ROC eğrisini kaydet
    plt.savefig(os.path.join(Config.STATIC_PATH, 'images', 'roc_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Model, scaler ve metrikleri kaydet
    print("Model, scaler ve metrikler kaydediliyor...")
    joblib.dump(best_model, Config.BEST_MODEL)
    joblib.dump(scaler, Config.SCALER)
    
    with open(Config.METRICS, 'w') as f:
        json.dump(metrics, f)
    
    print("Model, scaler ve metrikler başarıyla kaydedildi!")
    print("\nModel Performansı:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    train_and_save_model()