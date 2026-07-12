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


# =========================================================================
# BASIS ATURAN (RULE BASE) UNTUK FORWARD CHAINING
# =========================================================================
# Setiap aturan berbentuk IF (kondisi atas working memory) THEN (fakta baru
# ditambahkan ke working memory). Mesin inferensi mencocokkan aturan secara
# berulang (iteratif) terhadap working memory yang terus bertambah, sehingga
# kesimpulan dari satu aturan bisa memicu aturan lain (chaining) -- bukan
# sekadar pemetaan langsung 1 fitur -> 1 saran.

def _punya_penyebab(fitur):
    """Kondisi: True jika fakta ('penyebab', fitur) ada di working memory."""
    return lambda wm: ('penyebab', fitur) in wm


def _punya_pelindung(fitur):
    return lambda wm: ('pelindung', fitur) in wm


def _ada_kesimpulan(label):
    return lambda wm: ('kesimpulan', label) in wm


def _buat_aksi_saran(fitur):
    def aksi(wm):
        return {('saran', fitur)}
    return aksi


def _buat_aksi_apresiasi(fitur):
    def aksi(wm):
        return {('apresiasi', fitur)}
    return aksi


# LAYER 1: tiap fitur penyebab men-trigger fakta 'saran' untuk fitur itu,
# dan tiap fitur pelindung men-trigger fakta 'apresiasi'. Ini jadi pijakan
# bagi aturan lapis berikutnya (chaining).
RULES = []

for _fitur in SARAN_PENYEBAB:
    RULES.append({
        'nama': f'R_SARAN_{_fitur}',
        'if': _punya_penyebab(_fitur),
        'then': _buat_aksi_saran(_fitur),
    })

for _fitur in APRESIASI_PELINDUNG:
    RULES.append({
        'nama': f'R_APRESIASI_{_fitur}',
        'if': _punya_pelindung(_fitur),
        'then': _buat_aksi_apresiasi(_fitur),
    })


# LAYER 2: aturan gabungan yang baru bisa "fire" setelah fakta 'saran'
# individual sudah tersimpulkan lebih dulu (fakta hasil rule sebelumnya
# dipakai sebagai premis rule berikutnya -> chaining bertingkat).
# Contoh 1: kecemasan & beban belajar yang sama-sama jadi penyebab saling
# memperkuat, sehingga disimpulkan fakta baru 'pola_akademik_cemas'.
def _rule_pola_akademik_cemas(wm):
    return ('saran', 'anxiety_level') in wm and ('saran', 'study_load') in wm

def _aksi_pola_akademik_cemas(wm):
    return {('kesimpulan', 'pola_akademik_cemas')}

RULES.append({
    'nama': 'R_KESIMPULAN_pola_akademik_cemas',
    'if': _rule_pola_akademik_cemas,
    'then': _aksi_pola_akademik_cemas,
})

def _aksi_saran_pola_akademik_cemas(wm):
    return {('saran_tambahan', 'pola_akademik_cemas')}

RULES.append({
    'nama': 'R_SARAN_TAMBAHAN_pola_akademik_cemas',
    'if': _ada_kesimpulan('pola_akademik_cemas'),
    'then': _aksi_saran_pola_akademik_cemas,
})

# Contoh 2: dua atau lebih keluhan fisik (headache/blood_pressure/breathing)
# muncul bersamaan -> disimpulkan sebagai indikasi psikosomatis, memicu
# fakta baru dan saran tambahan yang tidak ada di basis pengetahuan dasar.
def _rule_psikosomatis(wm):
    gejala = [('saran', f) for f in ('headache', 'blood_pressure', 'breathing_problem')]
    return sum(1 for g in gejala if g in wm) >= 2

def _aksi_psikosomatis(wm):
    return {('kesimpulan', 'indikasi_psikosomatis')}

RULES.append({
    'nama': 'R_KESIMPULAN_psikosomatis',
    'if': _rule_psikosomatis,
    'then': _aksi_psikosomatis,
})

def _aksi_saran_psikosomatis(wm):
    return {('saran_tambahan', 'indikasi_psikosomatis')}

RULES.append({
    'nama': 'R_SARAN_TAMBAHAN_psikosomatis',
    'if': _ada_kesimpulan('indikasi_psikosomatis'),
    'then': _aksi_saran_psikosomatis,
})

# Teks saran tambahan yang dipicu oleh fakta hasil chaining (bukan lookup
# langsung dari fitur mentah, melainkan dari KESIMPULAN antara).
SARAN_TAMBAHAN = {
    'pola_akademik_cemas': (
        "Kombinasi kecemasan dan beban belajar yang tinggi saling memperberat satu sama lain. "
        "Pertimbangkan konsultasi dengan konselor akademik untuk menyusun strategi belajar sekaligus "
        "teknik manajemen kecemasan secara bersamaan, bukan ditangani terpisah."
    ),
    'indikasi_psikosomatis': (
        "Munculnya beberapa keluhan fisik sekaligus (sakit kepala, tekanan darah, dan/atau pernapasan) "
        "bisa jadi merupakan manifestasi psikosomatis dari stres. Selain penanganan medis untuk tiap "
        "gejala, disarankan pemeriksaan menyeluruh yang mengaitkan kondisi fisik dengan kondisi mental."
    ),
}


def _forward_chaining(working_memory, rules):
    """
    Mesin inferensi Forward Chaining generik.

    Bekerja secara iteratif: pada tiap putaran, semua aturan dicocokkan
    terhadap working memory saat ini. Fakta baru dari aturan yang "fire"
    dikumpulkan lalu ditambahkan ke working memory di akhir putaran.
    Proses berulang selama masih ada fakta baru yang dihasilkan, sehingga
    fakta hasil satu aturan bisa memicu aturan lain (chaining berlapis),
    dan berhenti otomatis saat mencapai fixpoint (tidak ada lagi aturan
    baru yang bisa fire).
    """
    wm = set(working_memory)
    fired_log = []

    while True:
        fakta_baru = set()
        for rule in rules:
            if rule['if'](wm):
                hasil_aksi = rule['then'](wm)
                tambahan = hasil_aksi - wm
                if tambahan:
                    fakta_baru |= tambahan
                    fired_log.append(rule['nama'])

        if not fakta_baru:
            break  # fixpoint tercapai, hentikan iterasi
        wm |= fakta_baru

    return wm, fired_log


def beri_saran(tingkat_stres, daftar_penyebab, daftar_pelindung):
    """
    Rule Based Reasoning dengan Forward Chaining sesungguhnya:
    1. Fakta awal (tingkat stres, penyebab dominan, faktor pelindung)
       dimasukkan ke working memory.
    2. Mesin inferensi mencocokkan basis aturan (RULES) secara iteratif
       terhadap working memory. Fakta baru hasil satu aturan (mis. fakta
       'saran' per fitur) dapat menjadi premis bagi aturan lapis
       berikutnya (mis. 'kesimpulan' pola gabungan), sampai tidak ada lagi
       aturan yang bisa dipicu (fixpoint) -> inilah proses chaining.
    3. Working memory akhir dibaca untuk menyusun output saran & apresiasi.
    """
    hasil = {
        'tingkat_stres': tingkat_stres,
        'saran_utama': [],
        'apresiasi': []
    }

    # RULE 1: Tidak Stres -> tidak perlu inferensi lebih lanjut
    if tingkat_stres == 'Tidak Stres':
        hasil['saran_utama'].append(
            "Kondisimu saat ini sangat baik. Pertahankan pola hidup sehatmu!"
        )
        return hasil

    # Fakta awal (initial facts) untuk working memory
    working_memory = set()
    working_memory.add(('tingkat_stres', tingkat_stres))
    for fitur in daftar_penyebab:
        working_memory.add(('penyebab', fitur))
    for fitur in daftar_pelindung:
        working_memory.add(('pelindung', fitur))

    # Jalankan mesin inferensi forward chaining sampai fixpoint
    wm_akhir, _fired_log = _forward_chaining(working_memory, RULES)

    # Susun saran utama berdasarkan urutan penyebab dominan (agar urutan
    # tampilan tetap mengikuti prioritas SHAP), diambil dari fakta 'saran'
    # yang benar-benar berhasil disimpulkan mesin inferensi.
    for fitur in daftar_penyebab:
        if ('saran', fitur) in wm_akhir:
            nama = NAMA_INDO.get(fitur, fitur)
            saran = SARAN_PENYEBAB[fitur]
            hasil['saran_utama'].append(f"[{nama}] {saran}")

    # Tambahkan saran hasil chaining lapis kedua (kesimpulan gabungan),
    # yang hanya muncul jika kombinasi fakta tertentu terpenuhi.
    for label, teks in SARAN_TAMBAHAN.items():
        if ('saran_tambahan', label) in wm_akhir:
            hasil['saran_utama'].append(f"[Pola Gabungan] {teks}")

    # Apresiasi untuk faktor pelindung, dari fakta yang disimpulkan mesin
    for fitur in daftar_pelindung:
        if ('apresiasi', fitur) in wm_akhir:
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