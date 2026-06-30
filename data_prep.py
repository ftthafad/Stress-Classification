import pandas as pd

# 1. Baca dataset
df = pd.read_csv('StressLevelDataset.csv')

# 2. Cek data kosong
data_kosong = df.isnull().sum().sum()

# 3. Kategorisasi Stress Level
def kategorikan_stress(nilai):
    if nilai == 0:
        return 'Tidak Stres'
    elif nilai == 1:
        return 'Stres Ringan'
    else:
        return 'Stres Berat'

df['stress_category'] = df['stress_level'].apply(kategorikan_stress)
df = df.drop(columns=['stress_level'])

# 4. Hitung distribusi
distribusi = df['stress_category'].value_counts()
total = len(df)

# 5. Tampilkan output
print("=== HASIL PREPROCESSING ===")
print(f"Jumlah data          : {total}")
print(f"Jumlah fitur         : {len(df.columns) - 1}")
print(f"Data kosong          : {data_kosong}")

print("\n=== DISTRIBUSI STRESS LEVEL ===")
for kategori, jumlah in distribusi.items():
    persentase = (jumlah / total) * 100
    print(f"{kategori:20s} : {jumlah} data ({persentase:.0f}%)")

# 6. Simpan
df.to_csv('processed-dataset.csv', index=False)
print("\nData berhasil disimpan ke processed-dataset.csv")