# reasoning.py
# Rule Based Reasoning untuk memberikan saran berdasarkan
# hasil prediksi Ensemble Learning + penyebab dominan dari SHAP
#
# Referensi jurnal per fitur:
# 1. Sleep Quality, Depression       -> Jurnal Psikologi Malahayati (2026)
# 2. Anxiety Level, Study Load       -> Jurnal Pendidikan dan Ilmu Sosial (2026)
# 3. Social Support                  -> Jurnal PAEDAGOGY
# 4. Peer Pressure                   -> Jurnal JOECY
# 5. Bullying                        -> Jurnal JPBIDKES
# 6. Self Esteem                     -> Jurnal Detector
# 7. Living Conditions, Noise Level  -> Jurnal Konseling dan Pendidikan
# 8. Future Career Concerns          -> Jurnal JBMED
# 9. Extracurricular Activities      -> Frontiers in Psychology (2024)
# 10. Headache, Blood Pressure,
#     Breathing Problem              -> Jurnal JRPP
# 11. Mental Health History,
#     Academic Performance,
#     Safety, Basic Needs            -> Jurnal Educate (2025)
# 12. Teacher Student Relationship   -> Jurnal Medika Saintika (2025)

# Basis pengetahuan: saran untuk tiap fitur sebagai PENYEBAB stres
SARAN_PENYEBAB = {
    'anxiety_level': "Lakukan teknik relaksasi seperti pernapasan dalam dan meditasi untuk mengelola kecemasan",
    'self_esteem': "Tingkatkan harga diri melalui pendekatan Cognitive Behavioral Therapy (CBT) dan latihan penerimaan diri",
    'mental_health_history': "Segera konsultasikan kondisi kesehatan mentalmu dengan psikolog atau konselor kampus",
    'depression': "Perbaiki kualitas tidur dan pertimbangkan konsultasi dengan psikolog untuk mengelola gejala depresi",
    'headache': "Konsultasikan keluhan sakit kepala dengan dokter dan kelola stres secara holistik",
    'blood_pressure': "Periksakan tekanan darahmu ke dokter dan kelola stres untuk menjaga kesehatan fisik",
    'sleep_quality': "Perbaiki kualitas tidur dengan menghindari gadget sebelum tidur dan menjaga jadwal tidur yang konsisten",
    'breathing_problem': "Konsultasikan masalah pernapasan dengan dokter dan kurangi aktivitas yang memicu stres berlebihan",
    'noise_level': "Cari lingkungan yang lebih tenang untuk belajar atau gunakan headphone peredam bising",
    'living_conditions': "Pertimbangkan untuk memperbaiki kondisi tempat tinggal dan manfaatkan layanan konseling kampus jika diperlukan",
    'safety': "Cari dukungan dari pihak kampus atau keluarga untuk meningkatkan rasa aman di lingkunganmu",
    'basic_needs': "Cari bantuan dari pihak kampus, keluarga, atau lembaga terkait untuk memenuhi kebutuhan dasarmu",
    'academic_performance': "Terapkan manajemen waktu yang lebih efektif dan cari dukungan akademik bila diperlukan",
    'study_load': "Buat jadwal belajar yang teratur dan bagi tugas menjadi bagian-bagian kecil agar lebih mudah dikelola",
    'teacher_student_relationship': "Tingkatkan komunikasi dengan dosen/guru dan manfaatkan layanan konseling kampus",
    'future_career_concerns': "Konsultasikan kekhawatiran kariermu dengan dosen pembimbing akademik atau orang tua",
    'social_support': "Tingkatkan interaksi dengan teman, keluarga, atau komunitas untuk mendapatkan dukungan emosional",
    'peer_pressure': "Ikuti konseling atau pelatihan manajemen stres untuk menguatkan keterampilan sosial dalam menghadapi tekanan kelompok",
    'extracurricular_activities': "Tingkatkan keaktifan dalam kegiatan ekstrakurikuler atau aktivitas fisik untuk membantu mengelola stres",
    'bullying': "Ikuti program peer support atau konseling untuk meningkatkan resiliensi dan kepercayaan diri",
}

# Apresiasi untuk fitur yang menjadi FAKTOR PELINDUNG (sudah baik)
APRESIASI_PELINDUNG = {
    'anxiety_level': "tingkat kecemasanmu yang terkendali",
    'self_esteem': "kepercayaan dirimu yang baik",
    'mental_health_history': "riwayat kesehatan mentalmu yang baik",
    'depression': "kondisi mentalmu yang stabil",
    'headache': "kondisi fisikmu yang sehat",
    'blood_pressure': "tekanan darahmu yang normal",
    'sleep_quality': "kualitas tidurmu yang baik",
    'breathing_problem': "kondisi pernapasanmu yang baik",
    'noise_level': "lingkunganmu yang tenang",
    'living_conditions': "kondisi tempat tinggalmu yang baik",
    'safety': "rasa amanmu di lingkungan sekitar",
    'basic_needs': "kebutuhan dasarmu yang terpenuhi",
    'academic_performance': "performa akademikmu yang baik",
    'study_load': "beban belajarmu yang terkelola dengan baik",
    'teacher_student_relationship': "hubunganmu yang baik dengan guru/dosen",
    'future_career_concerns': "kesiapanmu menghadapi masa depan karier",
    'social_support': "dukungan sosial yang kamu miliki",
    'peer_pressure': "ketahananmu terhadap tekanan teman sebaya",
    'extracurricular_activities': "keaktifanmu dalam kegiatan ekstrakurikuler",
    'bullying': "kondisimu yang terhindar dari perundungan",
}

NAMA_INDO = {
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
    'bullying': 'Perundungan',
}


def beri_saran(tingkat_stres, daftar_penyebab, daftar_pelindung):
    """
    Rule Based Reasoning (Forward Chaining):
    1. Cek tingkat stres sebagai fakta awal
    2. Jika Tidak Stres -> beri apresiasi, tidak perlu saran perbaikan
    3. Jika Stres Ringan/Berat -> cocokkan tiap penyebab dominan dengan
       basis pengetahuan SARAN_PENYEBAB, lalu tambahkan apresiasi
       untuk faktor pelindung yang dimiliki
    """
    hasil = {
        'tingkat_stres': tingkat_stres,
        'saran_utama': [],
        'apresiasi': []
    }

    # RULE 1: Tidak Stres
    if tingkat_stres == 'Tidak Stres':
        hasil['saran_utama'].append(
            "Kondisimu saat ini sangat baik. Pertahankan pola hidup sehatmu!"
        )
        return hasil

    # RULE 2: Stres Ringan atau Stres Berat
    # Cocokkan tiap fitur penyebab dengan basis pengetahuan
    for fitur in daftar_penyebab:
        if fitur in SARAN_PENYEBAB:
            nama = NAMA_INDO.get(fitur, fitur)
            saran = SARAN_PENYEBAB[fitur]
            hasil['saran_utama'].append(f"[{nama}] {saran}")

    # RULE 3: Tambahkan apresiasi untuk faktor pelindung
    for fitur in daftar_pelindung:
        if fitur in APRESIASI_PELINDUNG:
            hasil['apresiasi'].append(APRESIASI_PELINDUNG[fitur])

    return hasil


def tampilkan_hasil(hasil):
    print("\n=== SARAN (Rule Based Reasoning) ===")

    if hasil['tingkat_stres'] == 'Tidak Stres':
        print(hasil['saran_utama'][0])
        return

    print("\nBerdasarkan faktor penyebab dominan, berikut saran untukmu:")
    for i, saran in enumerate(hasil['saran_utama'], 1):
        print(f"{i}. {saran}")

    if hasil['apresiasi']:
        gabungan = ", ".join(hasil['apresiasi'])
        print(f"\nKamu juga memiliki kekuatan berupa {gabungan}.")
        print("Manfaatkan kekuatan ini untuk membantu menghadapi stres yang kamu alami.")


# Contoh pengujian mandiri
if __name__ == "__main__":
    contoh_penyebab = ['anxiety_level', 'study_load']
    contoh_pelindung = ['self_esteem', 'social_support']

    hasil = beri_saran('Stres Berat', contoh_penyebab, contoh_pelindung)
    tampilkan_hasil(hasil)