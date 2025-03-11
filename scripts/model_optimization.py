# Kredi Kartı Dolandırıcılık Tespiti - Model Optimizasyonu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
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

def plot_learning_curves(estimator, X, y, model_name):
    """Öğrenme eğrilerini çizer"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=train_sizes,
        scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Eğitim skoru', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation skoru', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    
    plt.xlabel('Eğitim Örnekleri Sayısı')
    plt.ylabel('F1 Skoru')
    plt.title(f'{model_name} Öğrenme Eğrileri')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'learning_curves_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, X_train, model_name):
    """Özellik önem analizini görselleştirir"""
    feature_importance = pd.DataFrame({
        'feature': [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'{model_name} - En Önemli 10 Özellik')
    plt.xlabel('Özellik Önemi')
    plt.ylabel('Özellik')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def optimize_models(X_train, X_test, y_train, y_test):
    """Modelleri optimize eder ve sonuçları görselleştirir"""
    
    # Model parametreleri
    param_grids = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 4],
                'subsample': [0.8, 1.0]
            }
        }
    }
    
    results = {}
    
    for name, model_info in param_grids.items():
        print(f"\n{name} için Grid Search başlıyor...")
        
        # Grid Search
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Sonuçları kaydet
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        # En iyi model ile tahminler
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Performans metrikleri
        print(f"\n{name} En İyi Model Performansı:")
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi F1 skoru: {grid_search.best_score_:.4f}")
        print(f"Test seti ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Öğrenme eğrileri
        plot_learning_curves(best_model, X_train, y_train, name)
        
        # Özellik önem analizi (sadece Random Forest için)
        if name == 'Random Forest':
            plot_feature_importance(best_model, X_train, name)
    
    return results

def main():
    # Veriyi yükle ve ön işle
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Modelleri optimize et
    results = optimize_models(X_train, X_test, y_train, y_test)
    
    # Sonuçları özetle
    print("\nOptimizasyon Sonuçları:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"En iyi parametreler: {result['best_params']}")
        print(f"En iyi F1 skoru: {result['best_score']:.4f}")

if __name__ == "__main__":
    main() 