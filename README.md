# Kredi Kartı Dolandırıcılık Tespit Sistemi

Bu proje, makine öğrenmesi kullanarak kredi kartı dolandırıcılıklarını tespit eden bir web uygulamasıdır.

## Önemli Not

Bu proje bir demo/prototiptir:
- Gösterilen veriler gerçek kredi kartı işlemlerini veya dolandırıcılık vakalarını temsil etmemektedir
- Dashboard'daki veriler, sistemin nasıl çalışacağını göstermek için üretilmiş örnek verilerdir
- Gerçek bir üretim ortamında, sistem gerçek banka/finans kurumu verileriyle entegre çalışacak şekilde düzenlenmelidir

## Özellikler

- Gerçek zamanlı dolandırıcılık tespiti
- Detaylı dashboard ve istatistikler
- Eğitim merkezi ve bilgilendirme sayfaları
- Teknik detaylar ve model performans metrikleri
- RESTful API desteği

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone https://github.com/Losdreyfedus/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

2. Veri setini indirin:
- [Credit Card Fraud Detection veri setini Kaggle'dan indirin](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- İndirilen `creditcard.csv` dosyasını `data/` klasörüne yerleştirin

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Modeli eğitin:
```bash
python scripts/save_best_model.py
```

5. Uygulamayı başlatın:
```bash
python run.py
```

## Veri Seti

Bu projede kullanılan veri seti, Kaggle'da bulunan [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) veri setidir. Veri seti boyut kısıtlamaları nedeniyle GitHub'a dahil edilmemiştir. Lütfen yukarıdaki bağlantıdan indirin.

## Dizin Yapısı

```
creditcard/
├── app/
│   ├── models/
│   ├── routes/
│   ├── static/
│   │   └── images/
│   └── templates/
├── data/
├── model/
├── scripts/
└── tests/
```

## API Kullanımı

```python
import requests

url = 'http://localhost:5000/api/predict'
data = {
    'amount': 100.0,
    'time': 0,
    'v1': 0.0,
    # ... diğer özellikler
}

response = requests.post(url, json=data)
print(response.json())
```

## 🎯 Proje Hedefleri

- Kredi kartı işlemlerindeki dolandırıcılık vakalarını tespit etme
- Farklı makine öğrenmesi modellerinin performansını karşılaştırma
- Dengesiz veri seti problemine çözüm üretme
- Hiperparametre optimizasyonu ile en iyi modeli belirleme

## 📊 Model Performansları

### Random Forest
- F1 Skoru: 0.9631
- ROC AUC: 0.9819
- En İyi Parametreler:
  - n_estimators: 200
  - max_depth: 20
  - min_samples_leaf: 1
  - min_samples_split: 2

### Gradient Boosting
- F1 Skoru: 0.9581
- ROC AUC: 0.9838
- En İyi Parametreler:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 4
  - subsample: 0.8

## 👤 Yazar

Hamza KAR