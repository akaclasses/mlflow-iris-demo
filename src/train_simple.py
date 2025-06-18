"""
Classification d'iris - Méthode classique (sans MLflow)
Le scientifique désorganisé qui ne note rien ! 🔬
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from datetime import datetime

print("🔬 Expérience ML classique - Méthode 'artisanale'")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Paramètres (cachés dans le code, pas documentés)
n_estimators = 15  # Pourquoi 15 ?
max_depth = 3      # Profondeur max des arbres, mais pourquoi 3 ?
test_size = 0.30   # Testé plusieurs valeurs, mais laquelle était la meilleure ?
random_seed = 42   # Toujours 42, mais pourquoi ?

print(f"🧪 Test avec {n_estimators} arbres, profondeur {max_depth}, split {test_size}")

# 1. Données (d'où viennent-elles ?)
iris = datasets.load_iris()  # Dataset iris intégré à scikit-learn
X, y = iris.data, iris.target
print(f"📊 Données: {len(X)} échantillons, {X.shape[1]} caractéristiques")
print(f"🎯 Classes: {iris.target_names}")

# 2. Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

# 3. Entraînement
model = RandomForestClassifier(
    n_estimators=n_estimators, 
    max_depth=max_depth,
    random_state=random_seed
)
model.fit(X_train, y_train)

# 4. Évaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 5. Sauvegarde "artisanale"
model_filename = f"iris_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Précision: {accuracy:.2%}")
print(f"💾 Modèle sauvé: {model_filename}")