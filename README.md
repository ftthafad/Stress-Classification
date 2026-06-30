# 🧠 Prediksi Tingkat Stres dan Analisis Penyebab Stres

> Menggunakan **Ensemble Learning** dan **Rule Based Learning**

Sistem prediksi tingkat stres berbasis machine learning yang menggabungkan tiga metode:
1. **Ensemble Learning** (Decision Tree + Random Forest + AdaBoost) untuk prediksi tingkat stres
2. **SHAP Analysis** untuk mengidentifikasi penyebab dominan stres
3. **Rule Based Reasoning** (Forward Chaining) untuk memberikan saran personal berbasis jurnal

---

## 📁 Struktur Proyek

```
Stress-Classification/
├── StressLevelDataset.csv       # Dataset mentah (1100 data, 20 fitur + 1 target)
├── data_prep.py                 # Preprocessing & kategorisasi stress level
├── train_model.py               # Training model Ensemble (VotingClassifier)
├── metode_shap.py               # Analisis SHAP (versi CLI)
├── reasoning.py                 # Rule Based Reasoning (basis pengetahuan + saran)
├── app.py                       # Program utama CLI (gabungan semua modul)
├── requirements.txt             # Dependencies Python
│
├── website/                     # 🌐 Website interaktif
│   ├── server.py                # Flask API server
│   └── static/
│       ├── index.html           # Halaman utama
│       ├── style.css            # Dark glassmorphism design
│       └── script.js            # Frontend logic (Vanilla JS)
│
├── processed-dataset.csv        # (generated) Dataset setelah preprocessing
└── model_ensemble.pkl           # (generated) Model ensemble yang sudah ditraining
```

---

## 🚀 Cara Menjalankan

### Prasyarat

- **Python 3.9+**
- **pip** (package manager Python)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies yang digunakan:
| Package | Fungsi |
|---------|--------|
| `flask` | Web server API |
| `pandas` | Manipulasi data |
| `scikit-learn` | Model machine learning (DT, RF, AdaBoost, VotingClassifier) |
| `shap` | Explainable AI — analisis kontribusi fitur |
| `numpy` | Komputasi numerik |

### 2. Preprocessing Dataset

```bash
python data_prep.py
```

Output:
- Mengubah `stress_level` (0, 1, 2) menjadi kategori: **Tidak Stres**, **Stres Ringan**, **Stres Berat**
- Menyimpan hasil ke `processed-dataset.csv`

### 3. Training Model

```bash
python train_model.py
```

Output:
- Melatih model Ensemble (Decision Tree + Random Forest + AdaBoost) dengan VotingClassifier (soft voting)
- Menampilkan akurasi masing-masing model dan ensemble
- Menjalankan 5-Fold Cross Validation
- Menyimpan model ke `model_ensemble.pkl`

### 4. Menjalankan Website

```bash
python website/server.py
```

Buka browser dan akses: **http://localhost:5001**

### 4b. Alternatif: Menjalankan via CLI

```bash
python app.py
```

---

## 🌐 Fitur Website

### Hero Section
Halaman pembuka dengan judul proyek dan navigasi ke setiap section.

### Metodologi
Penjelasan tiga pilar analisis: Ensemble Learning, SHAP Analysis, dan Rule Based Reasoning.

### Kuesioner
Form interaktif dengan **20 pertanyaan** menggunakan slider (skala 0–10), dikelompokkan menjadi:

| Kategori | Fitur |
|----------|-------|
| 🧠 **Psikologis** | Tingkat Kecemasan, Harga Diri, Riwayat Kesehatan Mental, Tingkat Depresi |
| 🏥 **Fisik & Kesehatan** | Sakit Kepala, Tekanan Darah, Kualitas Tidur, Masalah Pernapasan |
| 🏠 **Lingkungan** | Tingkat Kebisingan, Kondisi Tempat Tinggal, Rasa Aman, Kebutuhan Dasar |
| 📚 **Akademik & Sosial** | Performa Akademik, Beban Belajar, Hubungan dengan Guru, Kekhawatiran Karier, Dukungan Sosial, Tekanan Teman Sebaya, Kegiatan Ekstrakurikuler, Perundungan |

### Hasil Analisis
- **Tingkat Stres** — Prediksi dari model ensemble (Tidak Stres / Stres Ringan / Stres Berat)
- **SHAP Chart** — Visualisasi bar chart kontribusi setiap fitur (positif = meningkatkan stres, negatif = menurunkan stres)
- **Saran** — Rekomendasi personal dari Rule Based Reasoning berdasarkan penyebab dominan
- **Apresiasi** — Penghargaan untuk faktor pelindung yang dimiliki pengguna

---

## 🔬 Metodologi

### Ensemble Learning
Menggunakan `VotingClassifier` (soft voting) dari scikit-learn dengan tiga model base:
- **Decision Tree** (criterion: entropy)
- **Random Forest** (100 estimators)
- **AdaBoost**

### SHAP (SHapley Additive exPlanations)
- Menggunakan `KernelExplainer` untuk menjelaskan prediksi model ensemble
- Fitur dengan nilai SHAP **positif** = penyebab dominan yang **meningkatkan** stres
- Fitur dengan nilai SHAP **negatif** = faktor pelindung yang **menurunkan** stres

### Rule Based Reasoning
- Menggunakan **Forward Chaining** untuk mencocokkan penyebab dominan dengan basis pengetahuan
- Saran diambil dari referensi **jurnal-jurnal terpercaya** (lihat `reasoning.py` untuk daftar jurnal)
- Memberikan apresiasi untuk faktor pelindung yang sudah baik

---

## 📊 Dataset

- **Sumber**: `StressLevelDataset.csv`
- **Jumlah Data**: 1100
- **Jumlah Fitur**: 20
- **Target**: `stress_level` → dikategorikan menjadi 3 kelas

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Tidak Stres | 373 | 34% |
| Stres Ringan | 358 | 33% |
| Stres Berat | 369 | 34% |

---

## 📂 Penjelasan File

| File | Deskripsi |
|------|-----------|
| `data_prep.py` | Preprocessing: cek missing values, kategorisasi stress level, simpan dataset bersih |
| `train_model.py` | Training model ensemble, evaluasi akurasi, 5-fold cross validation |
| `metode_shap.py` | Analisis SHAP versi CLI (standalone) |
| `reasoning.py` | Basis pengetahuan Rule Based Reasoning + fungsi `beri_saran()` |
| `app.py` | Program utama CLI yang menggabungkan semua modul |
| `website/server.py` | Flask API server yang menghubungkan frontend ke model |
| `website/static/index.html` | Halaman utama website |
| `website/static/style.css` | Styling (dark glassmorphism theme) |
| `website/static/script.js` | Frontend logic (Vanilla JavaScript) |
