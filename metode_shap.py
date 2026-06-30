import pandas as pd
import pickle
import shap
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load model ensemble (VotingClassifier: Decision Tree + Random Forest + AdaBoost)
with open('model_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Definisi pertanyaan
PERTANYAAN = [
    ('anxiety_level', 'Seberapa sering kamu merasa cemas?', '0=tidak pernah, 10=sangat sering', 0, 21),
    ('self_esteem', 'Seberapa tinggi kepercayaan dirimu?', '0=sangat rendah, 10=sangat tinggi', 0, 30),
    ('mental_health_history', 'Apakah kamu pernah didiagnosis gangguan mental?', '0=tidak, 1=pernah', 0, 1),
    ('depression', 'Seberapa sering kamu merasa sedih atau putus asa?', '0=tidak pernah, 10=sangat sering', 0, 27),
    ('headache', 'Seberapa sering kamu mengalami sakit kepala?', '0=tidak pernah, 10=sangat sering', 0, 5),
    ('blood_pressure', 'Bagaimana kondisi tekanan darahmu?', '0=normal, 10=sangat tinggi', 1, 3),
    ('sleep_quality', 'Bagaimana kualitas tidurmu?', '0=sangat buruk, 10=sangat baik', 0, 5),
    ('breathing_problem', 'Seberapa sering kamu mengalami masalah pernapasan?', '0=tidak pernah, 10=sangat sering', 0, 5),
    ('noise_level', 'Seberapa bising lingkungan sekitarmu?', '0=sangat sepi, 10=sangat bising', 0, 5),
    ('living_conditions', 'Bagaimana kondisi tempat tinggalmu?', '0=sangat buruk, 10=sangat baik', 0, 5),
    ('safety', 'Seberapa aman kamu merasa di lingkunganmu?', '0=tidak aman, 10=sangat aman', 0, 5),
    ('basic_needs', 'Apakah kebutuhan pokokmu (makan, minum, tempat tinggal) terpenuhi?', '0=tidak terpenuhi, 10=sangat terpenuhi', 0, 5),
    ('academic_performance', 'Bagaimana performa akademikmu saat ini?', '0=sangat buruk, 10=sangat baik', 0, 5),
    ('study_load', 'Seberapa berat beban belajarmu?', '0=sangat ringan, 10=sangat berat', 0, 5),
    ('teacher_student_relationship', 'Bagaimana hubunganmu dengan guru/dosen?', '0=sangat buruk, 10=sangat baik', 0, 5),
    ('future_career_concerns', 'Seberapa khawatir kamu tentang karier masa depanmu?', '0=tidak khawatir, 10=sangat khawatir', 0, 5),
    ('social_support', 'Seberapa besar dukungan sosial yang kamu terima?', '0=tidak ada, 10=sangat besar', 0, 3),
    ('peer_pressure', 'Seberapa besar tekanan yang kamu rasakan dari teman sebaya?', '0=tidak ada, 10=sangat besar', 0, 5),
    ('extracurricular_activities', 'Seberapa aktif kamu mengikuti kegiatan ekstrakurikuler?', '0=tidak aktif, 10=sangat aktif', 0, 5),
    ('bullying', 'Seberapa sering kamu mengalami perundungan?', '0=tidak pernah, 10=sangat sering', 0, 5),
]


def skala_ke_asli(nilai_0_10, min_asli, max_asli):
    """Konversi nilai input pengguna (skala 0-10) ke skala asli fitur dataset."""
    asli = min_asli + (nilai_0_10 / 10) * (max_asli - min_asli)
    return int(asli + 0.5)


# 3. Input data dari pengguna
print("=== INPUT DATA KAMU (skala 0-10) ===\n")

data_skala_10 = {}
for key, pertanyaan, keterangan, _, _ in PERTANYAAN:
    while True:
        try:
            nilai = int(input(f"{pertanyaan} ({keterangan}): "))
            if key == 'mental_health_history':
                if nilai in [0, 1]:
                    break
                print("  -> Masukkan 0 atau 1 saja.")
            else:
                if 0 <= nilai <= 10:
                    break
                print("  -> Masukkan angka antara 0 sampai 10.")
        except ValueError:
            print("  -> Masukkan angka yang valid.")
    data_skala_10[key] = nilai

# 4. Konversi skala 0-10 -> skala asli
data = {}
for key, _, _, min_asli, max_asli in PERTANYAAN:
    if key == 'mental_health_history':
        data[key] = data_skala_10[key]
    else:
        data[key] = skala_ke_asli(data_skala_10[key], min_asli, max_asli)

print("\n=== KONVERSI INPUT (skala 0-10 -> skala asli model) ===")
for key, _, _, min_asli, max_asli in PERTANYAAN:
    print(f"{key:30s} : {data_skala_10[key]:2d}/10  ->  {data[key]} (rentang asli {min_asli}-{max_asli})")

# 5. Ubah ke dataframe
input_df = pd.DataFrame([data])

# 6. Prediksi stress level dengan model ensemble
prediksi = model.predict(input_df)[0]

# 7. Hitung SHAP
df_train = pd.read_csv('processed-dataset.csv')
X_train = df_train.drop(columns=['stress_category'])

print("\nMenghitung SHAP value untuk model ensemble (DT + RF + AdaBoost)...")
print("Proses ini bisa memakan waktu lebih lama dibanding model tunggal, mohon tunggu.\n")

explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100, random_state=42))
shap_values = explainer.shap_values(input_df, nsamples=100)

# 8. Ambil nilai SHAP
kelas = list(model.classes_)
idx_kelas = kelas.index(prediksi)

if isinstance(shap_values, list):
    nilai_shap = shap_values[idx_kelas][0]
else:
    if len(shap_values.shape) == 3:
        nilai_shap = shap_values[0, :, idx_kelas]
    else:
        nilai_shap = shap_values[0]

# 9. Buat dataframe SHAP
shap_df = pd.DataFrame({
    'Fitur': input_df.columns.tolist(),
    'Nilai SHAP': nilai_shap
}).sort_values('Nilai SHAP', ascending=False)

# 10. Pisah fitur meningkatkan dan menurunkan stres
meningkatkan = shap_df[shap_df['Nilai SHAP'] > 0].head(5)
menurunkan = shap_df[shap_df['Nilai SHAP'] < 0].tail(5)

# 11. Nama fitur dalam bahasa indonesia
nama_indo = {
    'anxiety_level': 'Tingkat Kecemasan',
    'self_esteem': 'Harga Diri',
    'mental_health_history': 'Riwayat Kesehatan Mental',
    'depression': 'Tingkat Depresi',
    'headache': 'Sakit Kepala',
    'blood_pressure': 'Tekanan Darah',
    'sleep_quality': 'Kualitas Tidur',
    'breathing_problem': 'Masalah Pernapasan',
    'noise_level': 'Tingkat Kebisingan',
    'living_conditions': 'Kondisi Tempat Tinggal',
    'safety': 'Rasa Aman',
    'basic_needs': 'Kebutuhan Dasar',
    'academic_performance': 'Performa Akademik',
    'study_load': 'Beban Belajar',
    'teacher_student_relationship': 'Hubungan dengan Guru',
    'future_career_concerns': 'Kekhawatiran Karier',
    'social_support': 'Dukungan Sosial',
    'peer_pressure': 'Tekanan Teman Sebaya',
    'extracurricular_activities': 'Kegiatan Ekstrakurikuler',
    'bullying': 'Perundungan'
}

# 12. Tampilkan output
print("\n=== HASIL PREDIKSI (Model Ensemble: DT + RF + AdaBoost) ===")
print(f"Tingkat Stres        : {prediksi}")

print("\n=== PENYEBAB DOMINAN (SHAP) ===")

print("\nFitur yang MENINGKATKAN stres:")
for i, (_, row) in enumerate(meningkatkan.iterrows(), 1):
    nama = nama_indo.get(row['Fitur'], row['Fitur'])
    print(f"{i}. {nama:35s} : +{row['Nilai SHAP']:.4f}")

print("\nFitur yang MENURUNKAN stres:")
menurunkan_sorted = menurunkan.sort_values('Nilai SHAP')
for i, (_, row) in enumerate(menurunkan_sorted.iterrows(), 1):
    nama = nama_indo.get(row['Fitur'], row['Fitur'])
    print(f"{i}. {nama:35s} : {row['Nilai SHAP']:.4f}")