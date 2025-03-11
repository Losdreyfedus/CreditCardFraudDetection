# Kredi Kartı Dolandırıcılık Tespiti Projesi
# Gerekli kütüphaneleri import edelim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('default')
sns.set_theme()

def load_data(sample_size=10000):
    """Veri setini yükler ve örnekleme yapar"""
    print("Veri seti yükleniyor...")
    df = pd.read_csv('creditcard.csv')
    
    # Dolandırıcılık vakalarını al
    fraud_df = df[df['Class'] == 1]
    # Normal işlemlerden örnekleme yap
    normal_df = df[df['Class'] == 0].sample(n=len(fraud_df) * 2, random_state=42)
    
    # Verileri birleştir
    df_sample = pd.concat([fraud_df, normal_df])
    
    print("\nÖrnek veri seti boyutu:", df_sample.shape)
    print("\nSınıf dağılımı:")
    print(df_sample['Class'].value_counts(normalize=True))
    
    return df_sample

def analyze_data(df):
    """Veri seti hakkında detaylı analiz yapar"""
    print("\nVeri seti bilgileri:")
    print(df.info())
    
    # Sınıf dağılımı görselleştirmesi
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Class')
    plt.title('Dolandırıcılık ve Normal İşlem Dağılımı')
    plt.xlabel('Sınıf (0: Normal, 1: Dolandırıcılık)')
    plt.ylabel('İşlem Sayısı')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Pasta grafik
    plt.figure(figsize=(8, 8))
    class_dist = df['Class'].value_counts(normalize=True)
    plt.pie(class_dist, labels=['Normal', 'Dolandırıcılık'], 
            autopct='%1.1f%%', colors=['lightblue', 'red'])
    plt.title('Sınıf Dağılımı Yüzdesi')
    plt.savefig('class_distribution_pie.png')
    plt.close()

def preprocess_data(df):
    """Veri ön işleme yapar"""
    print("Veri ön işleme başlıyor...")
    
    # Özellikler ve hedef değişkeni ayır
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Veriyi eğitim ve test setlerine böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE ile dengesiz veri problemini çöz
    print("SMOTE uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print("Veri ön işleme tamamlandı.")
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Model eğitimi ve değerlendirmesi yapar"""
    print("\nModel eğitimi başlıyor...")
    
    # Random Forest modelini eğit
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Ağaç sayısını azalttık
    rf_model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = rf_model.predict(X_test)
    
    # Model performansını değerlendir
    print("\nModel Performans Metrikleri:")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix görselleştirmesi
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC eğrisi
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return rf_model

def main():
    # Veri setini yükle (daha küçük örnek)
    df = load_data()
    
    # Veri analizi yap
    analyze_data(df)
    
    # Veri ön işleme
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Model eğitimi ve değerlendirmesi
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 