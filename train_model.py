import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# 1. Baca processed dataset
df = pd.read_csv('processed-dataset.csv')

# 2. Pisah fitur (X) dan target (y)
X = df.drop(columns=['stress_category'])
y = df['stress_category']

# 3. Bagi data 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Inisialisasi model-model base
model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_ada = AdaBoostClassifier(random_state=42)

# 5. Gabungkan dalam VotingClassifier (Ensemble)
ensemble_model = VotingClassifier(
    estimators=[
        ('dt', model_dt),
        ('rf', model_rf),
        ('ada', model_ada)
    ],
    voting='soft'
)

# 6. Training model ensemble
ensemble_model.fit(X_train, y_train)

# 7. Hitung akurasi model ensemble dan model individu untuk perbandingan
akurasi_ensemble = ensemble_model.score(X_test, y_test)

# Train individual models just to show accuracy comparison
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_ada.fit(X_train, y_train)

akurasi_dt = model_dt.score(X_test, y_test)
akurasi_rf = model_rf.score(X_test, y_test)
akurasi_ada = model_ada.score(X_test, y_test)

# 8. 5-Fold Cross Validation untuk Ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = cross_val_score(ensemble_model, X, y, cv=skf)

# 9. Tampilkan output
print("=== HASIL TRAINING & PERBANDINGAN AKURASI ===")
print(f"Data Training             : {len(X_train)} data")
print(f"Data Testing              : {len(X_test)} data")
print(f"Akurasi Decision Tree     : {akurasi_dt * 100:.2f}%")
print(f"Akurasi Random Forest     : {akurasi_rf * 100:.2f}%")
print(f"Akurasi AdaBoost          : {akurasi_ada * 100:.2f}%")
print(f"Akurasi Model Ensemble    : {akurasi_ensemble * 100:.2f}%")

print("\n=== 5-FOLD CROSS VALIDATION (ENSEMBLE) ===")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1}                    : {score * 100:.2f}%")
print(f"Rata-rata                 : {fold_scores.mean() * 100:.2f}%")

# 10. Simpan model ensemble
with open('model_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

print("\nModel ensemble disimpan ke: model_ensemble.pkl")