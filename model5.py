import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Gerekli Kütüphaneleri Yükleme ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

# Ayarlar ve Görüntüleme
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
warnings.filterwarnings('ignore')


# --- Merkezi Ön İşleme Fonksiyonu (SADELEŞTİRİLDİ) ---
def preprocess_data(df, base_columns_info, medians):
    """Veri setine temel ön işleme adımlarını tutarlı bir şekilde uygular."""
    processed_df = df.copy()
    for col in processed_df.columns:
        if col != 'dokum_no':
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    missing_cols = set(base_columns_info) - set(processed_df.columns)
    for col in missing_cols:
        processed_df[col] = medians[col]

    processed_df.fillna(medians, inplace=True)
    return processed_df


# --- 1. Veri Yükleme ve Ön İşleme ---
print("--- 1. Veri Yükleme ve Ön İşleme ---")
try:
    df_raw = pd.read_csv('final_birlesik_veri2.csv', sep=';', decimal=',')
except FileNotFoundError:
    print("Hata: 'final_birlesik_veri2.csv' bulunamadı.")
    exit()

# Yinelenen sütunları en başta kaldır
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
df_clean = df_raw.drop(columns=['dokum_no', 'numune_no'], errors='ignore')
numeric_df = df_clean.apply(pd.to_numeric, errors='coerce')
training_medians = numeric_df.median()
joblib.dump(training_medians, 'training_medians.joblib')
print("Ön işleme tamamlandı.")

# --- 2. Veri Hazırlama ---
print("\n--- 2. Veri Hazırlama ---")
target_columns = ['cap', 'ovalite', 'elastikiyet', 'rel_alt_akma_dayanimi', 'reh_ust_akma_dayanimi', 'tufal_orani']
feature_columns = [col for col in df_clean.columns if col not in target_columns]
X = df_clean[feature_columns]
y = df_clean[target_columns]
joblib.dump(feature_columns, 'feature_columns.joblib')

# Hedef değişkenlerin dağılımını normale yaklaştırmak için logaritmik dönüşüm
y_transformed = np.log1p(y)

# Veriyi Eğitim ve Test olarak ayır
X_train, X_test, y_train_transformed, y_test_transformed = train_test_split(X, y_transformed, test_size=0.2,
                                                                            random_state=42)
y_test_original = y.loc[y_test_transformed.index]

# Veriyi 0-1 aralığına normalize et
scaler = MinMaxScaler()
# scaler'ı sadece eğitim verisi üzerinde fit et ve kaydet
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'minmax_scaler.joblib')
print("Veri ayırma, hedef dönüşümü ve normalizasyon tamamlandı.")

# --- 3. Modellerin Optimizasyonu ve Eğitimi ---
print("\n" + "=" * 50 + "\n--- 3. Modellerin Optimizasyonu ve Eğitimi ---\n" + "=" * 50)
models_to_train = {
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}
trained_models = {}
model_results = []

# Her model için optimize edilecek parametre aralıkları
param_dist = {
    'n_estimators': randint(200, 1000),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.6, 0.4)
}

for name, model in models_to_train.items():
    print(f"\n--- {name} Modeli Eğitiliyor ve Optimize Ediliyor ---")
    target_specific_models = {}
    for target in target_columns:
        print(f"'{target}' için hiperparametre optimizasyonu başlatılıyor...")
        y_train_single_target = y_train_transformed[target]
        search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=10, cv=3,
            scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train_single_target)
        target_specific_models[target] = search.best_estimator_

    trained_models[name] = target_specific_models

    # Değerlendirme
    predictions_dict = {}
    for target, trained_model in target_specific_models.items():
        pred_transformed = trained_model.predict(X_test_scaled)
        pred_original = np.expm1(pred_transformed)
        predictions_dict[target] = pred_original
    y_pred_original_df = pd.DataFrame(predictions_dict, index=y_test_original.index)
    r2 = r2_score(y_test_original, y_pred_original_df)
    model_results.append({'Model': name, 'Ortalama_R2_Score': r2})
    print(f"{name} modeli değerlendirildi. Ortalama R2 Skoru: {r2:.4f}")

joblib.dump(trained_models, 'model_portfolio_final.pkl')

# --- 4. Model Karşılaştırma ve Hata Analizi ---
results_df = pd.DataFrame(model_results).sort_values(by='Ortalama_R2_Score', ascending=False)
print("\n" + "=" * 50 + "\n--- 4. Final Model Karşılaştırma Sonuçları ---\n" + "=" * 50)
print(results_df)

# --- 5. Test Seti Tahmin-Gerçek Karşılaştırması ve Görselleştirme ---
print("\n" + "=" * 50 + f"\n--- 5. Test Seti Analizi ve Görselleştirme ---\n" + "=" * 50)
for model_name, models in trained_models.items():
    print(f"\n--- Model: {model_name} ---")

    # Tahminleri oluştur
    predictions = {}
    for target, model in models.items():
        pred_transformed = model.predict(X_test_scaled)
        predictions[target] = np.expm1(pred_transformed)
    pred_df = pd.DataFrame(predictions, index=y_test_original.index)

    # Karşılaştırma tablosu
    comparison_df = y_test_original.copy()
    for target in target_columns:
        comparison_df[f'pred_{target}'] = pred_df[target]
    print(f"\n{model_name} - Test Seti Tahmin ve Gerçek Değer Karşılaştırması (İlk 10):")
    print(comparison_df.head(10).round(3))

    # Hata grafikleri
    for target_name in ['elastikiyet', 'rel_alt_akma_dayanimi']:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test_original[target_name], y=pred_df[target_name])
        plt.title(f'{model_name} - {target_name.capitalize()} - Gerçek vs. Tahmin')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.plot([y_test_original[target_name].min(), y_test_original[target_name].max()],
                 [y_test_original[target_name].min(), y_test_original[target_name].max()],
                 'r--')
        plt.savefig(f'{model_name}_tahmin_hata_grafigi_{target_name}.png')
        print(
            f"'{target_name}' için tahmin hata grafiği kaydedildi: {model_name}_tahmin_hata_grafigi_{target_name}.png")

# --- 6. Yeni Veri ile Tahmin Yapma ---
print("\n" + "=" * 50 + f"\n--- 6. Yeni Veri ile Tahmin ---\n" + "=" * 50)
try:
    new_data_raw = pd.read_csv('yeni_dokum_veri.csv', sep=';', decimal=',')
    print("'yeni_dokum_veri.csv' dosyası başarıyla okundu.")

    new_data_raw = new_data_raw.loc[:, ~new_data_raw.columns.duplicated()]

    # Yeni veriye de aynı basit ön işlemeyi uygula
    new_data_processed = preprocess_data(new_data_raw, base_columns_info=feature_columns, medians=training_medians)

    # Sütunları eğitimdeki sırayla hizala ve normalize et
    new_data_aligned = new_data_processed[feature_columns]
    new_data_scaled = scaler.transform(new_data_aligned)

    # Her model ailesi için tahmin yap
    for model_name, models in trained_models.items():
        print(f"\n>>> {model_name.upper()} MODELİ İLE YENİ VERİ TAHMİNLERİ (İlk 5):")
        final_predictions_dict = {}
        for target, model in models.items():
            new_pred_transformed = model.predict(new_data_scaled)
            final_predictions_dict[f'pred_{target}'] = np.expm1(new_pred_transformed)
        final_predictions_df = pd.DataFrame(final_predictions_dict)
        print(final_predictions_df.head(5).round(3))

except FileNotFoundError:
    print("\nUYARI: 'yeni_dokum_veri.csv' bulunamadı.")
except Exception as e:
    print(f"\nYeni veri ile tahmin sırasında hata: {e}")