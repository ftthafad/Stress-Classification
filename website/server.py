# server.py
# Flask API server untuk website Prediksi Tingkat Stres
# Menyediakan endpoint untuk prediksi menggunakan Ensemble Learning,
# analisis SHAP, dan Rule Based Reasoning.

import os
import sys
import pandas as pd
import pickle
import shap
import numpy as np
import warnings
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

# Tambahkan parent directory ke sys.path agar bisa import reasoning.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from flask import Flask, request, jsonify, send_from_directory
from reasoning import beri_saran, NAMA_INDO

app = Flask(__name__, static_folder='static')

# ======================================================
# LOAD MODEL & DATA SAAT SERVER START
# ======================================================
MODEL_PATH = os.path.join(ROOT_DIR, 'model_ensemble.pkl')
DATA_PATH = os.path.join(ROOT_DIR, 'processed-dataset.csv')

model = None
X_train = None

def load_model():
    global model, X_train
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        df_train = pd.read_csv(DATA_PATH)
        X_train = df_train.drop(columns=['stress_category'])
        print("✅ Model dan dataset berhasil dimuat.")
    else:
        print("⚠️  File model atau dataset tidak ditemukan.")
        print(f"   Pastikan '{MODEL_PATH}' dan '{DATA_PATH}' tersedia.")
        print("   Jalankan data_prep.py lalu train_model.py terlebih dahulu.")

# ======================================================
# DEFINISI PERTANYAAN (sama dengan app.py)
# ======================================================
PERTANYAAN = [
    ('anxiety_level', 0, 21),
    ('self_esteem', 0, 30),
    ('mental_health_history', 0, 1),
    ('depression', 0, 27),
    ('headache', 0, 5),
    ('blood_pressure', 1, 3),
    ('sleep_quality', 0, 5),
    ('breathing_problem', 0, 5),
    ('noise_level', 0, 5),
    ('living_conditions', 0, 5),
    ('safety', 0, 5),
    ('basic_needs', 0, 5),
    ('academic_performance', 0, 5),
    ('study_load', 0, 5),
    ('teacher_student_relationship', 0, 5),
    ('future_career_concerns', 0, 5),
    ('social_support', 0, 3),
    ('peer_pressure', 0, 5),
    ('extracurricular_activities', 0, 5),
    ('bullying', 0, 5),
]


def skala_ke_asli(nilai_0_10, min_asli, max_asli):
    """Konversi nilai input pengguna (skala 0-10) ke skala asli fitur dataset."""
    asli = min_asli + (nilai_0_10 / 10) * (max_asli - min_asli)
    return int(asli + 0.5)


# ======================================================
# ROUTES
# ======================================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi stress.
    Input: JSON { "anxiety_level": 7, "self_esteem": 3, ... } (skala 0-10)
    Output: JSON { "prediksi", "penyebab", "pelindung", "saran", "apresiasi", "shap_values" }
    """
    if model is None or X_train is None:
        return jsonify({'error': 'Model belum dimuat. Jalankan data_prep.py dan train_model.py terlebih dahulu.'}), 500

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'Data input tidak valid.'}), 400

        # 1. Konversi skala 0-10 ke skala asli dataset
        data = {}
        data_skala_10 = {}
        for key, min_asli, max_asli in PERTANYAAN:
            if key not in input_data:
                return jsonify({'error': f'Fitur "{key}" tidak ditemukan dalam input.'}), 400

            nilai = int(input_data[key])
            data_skala_10[key] = nilai

            if key == 'mental_health_history':
                data[key] = nilai
            else:
                data[key] = skala_ke_asli(nilai, min_asli, max_asli)

        # 2. Prediksi menggunakan model ensemble
        input_df = pd.DataFrame([data])
        prediksi = model.predict(input_df)[0]

        # 3. Jika tidak stres, langsung return tanpa SHAP
        if prediksi == 'Tidak Stres':
            hasil = beri_saran(prediksi, [], [])
            return jsonify({
                'prediksi': prediksi,
                'penyebab': [],
                'pelindung': [],
                'shap_values': [],
                'saran': hasil['saran_utama'],
                'apresiasi': hasil['apresiasi']
            })

        # 4. Analisis SHAP
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_train, 100, random_state=42)
        )
        shap_values = explainer.shap_values(input_df, nsamples=100)

        kelas = list(model.classes_)
        idx_kelas = kelas.index(prediksi)

        if isinstance(shap_values, list):
            nilai_shap = shap_values[idx_kelas][0]
        else:
            if len(shap_values.shape) == 3:
                nilai_shap = shap_values[0, :, idx_kelas]
            else:
                nilai_shap = shap_values[0]

        shap_df = pd.DataFrame({
            'Fitur': input_df.columns.tolist(),
            'Nilai_SHAP': nilai_shap
        }).sort_values('Nilai_SHAP', ascending=False)

        # Penyebab dominan (SHAP > 0)
        daftar_penyebab = shap_df[shap_df['Nilai_SHAP'] > 0]['Fitur'].head(5).tolist()
        penyebab_data = []
        for _, row in shap_df[shap_df['Nilai_SHAP'] > 0].head(5).iterrows():
            penyebab_data.append({
                'fitur': row['Fitur'],
                'nama': NAMA_INDO.get(row['Fitur'], row['Fitur']),
                'nilai_shap': float(row['Nilai_SHAP'])
            })

        # Faktor pelindung (SHAP < 0)
        pelindung_sorted = shap_df[shap_df['Nilai_SHAP'] < 0].tail(5).sort_values('Nilai_SHAP')
        daftar_pelindung = pelindung_sorted['Fitur'].tolist()
        pelindung_data = []
        for _, row in pelindung_sorted.iterrows():
            pelindung_data.append({
                'fitur': row['Fitur'],
                'nama': NAMA_INDO.get(row['Fitur'], row['Fitur']),
                'nilai_shap': float(row['Nilai_SHAP'])
            })

        # 5. Rule-based reasoning
        hasil = beri_saran(prediksi, daftar_penyebab, daftar_pelindung)

        # Semua SHAP values untuk chart
        all_shap = []
        for _, row in shap_df.iterrows():
            all_shap.append({
                'fitur': row['Fitur'],
                'nama': NAMA_INDO.get(row['Fitur'], row['Fitur']),
                'nilai_shap': float(row['Nilai_SHAP'])
            })

        return jsonify({
            'prediksi': prediksi,
            'penyebab': penyebab_data,
            'pelindung': pelindung_data,
            'shap_values': all_shap,
            'saran': hasil['saran_utama'],
            'apresiasi': hasil['apresiasi']
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001)
