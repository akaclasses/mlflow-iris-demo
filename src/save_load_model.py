"""
Sauvegarde et chargement de modèles avec MLflow
Version simple pour apprentissage
"""
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("1️⃣ Entraînement et sauvegarde du modèle...")

# Préparer les données
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder avec MLflow
model_path = "iris_model"
mlflow.sklearn.save_model(model, model_path)
print(f"✅ Modèle sauvegardé dans: {model_path}")

print("\n2️⃣ Chargement et utilisation du modèle...")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

# Tester avec de nouvelles données
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Exemple de fleur Iris
prediction = loaded_model.predict(new_data)
proba = loaded_model.predict_proba(new_data)

print(f"✅ Modèle chargé avec succès")
print(f"🔮 Prédiction: {iris.target_names[prediction[0]]}")
print(f"📊 Probabilités: {proba[0]}")