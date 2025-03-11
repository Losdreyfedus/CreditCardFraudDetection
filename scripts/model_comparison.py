# Kredi Kartı Dolandırıcılık Tespiti - Model Karşılaştırması
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Veri setini yükler ve ön işler"""
    print("Veri seti yükleniyor...")
    df = pd.read_csv('creditcard.csv')
    
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
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def compare_models(X_train, X_test, y_train, y_test):
    """Farklı modelleri karşılaştırır"""
    models = {
        'Lojistik Regresyon': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    results = {}
    
    # ROC eğrileri için
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\n{name} modeli eğitiliyor...")
        model.fit(X_train, y_train)
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Performans raporu
        print(f"\n{name} Performans Metrikleri:")
        print(classification_report(y_test, y_pred))
        
        # ROC eğrisi
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        # Sonuçları sakla
        results[name] = {
            'auc': roc_auc,
            'accuracy': (y_pred == y_test).mean(),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    # ROC grafiğini tamamla
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrileri Karşılaştırması')
    plt.legend(loc="lower right")
    plt.savefig('model_comparison_roc.png')
    plt.close()
    
    # Model karşılaştırma grafiği
    metrics = pd.DataFrame(
        {name: {'AUC': results[name]['auc'], 
                'Accuracy': results[name]['accuracy']} 
         for name in models.keys()}).T
    
    plt.figure(figsize=(12, 6))
    metrics.plot(kind='bar', width=0.8)
    plt.title('Model Performans Karşılaştırması')
    plt.ylabel('Skor')
    plt.xlabel('Model')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png')
    plt.close()
    
    return results

def main():
    # Veriyi yükle ve ön işle
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Modelleri karşılaştır
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # En iyi modeli göster
    best_model = max(results.items(), key=lambda x: x[1]['auc'])[0]
    print(f"\nEn iyi model: {best_model}")
    print(f"AUC: {results[best_model]['auc']:.4f}")
    print(f"Accuracy: {results[best_model]['accuracy']:.4f}")

if __name__ == "__main__":
    main() 