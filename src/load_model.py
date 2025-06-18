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

print("\n2️⃣ Chargement et utilisation du modèle...")

# Préparer les données
iris = datasets.load_iris()

# Charger le modèle
model_path = "iris_model"
loaded_model = mlflow.sklearn.load_model(model_path)

# Tester avec de nouvelles données
new_data = np.array([[1.1, 1.5, 4.4, 1.2]])  # Exemple de fleur Iris
prediction = loaded_model.predict(new_data)
proba = loaded_model.predict_proba(new_data)

print(f"✅ Modèle chargé avec succès")
print(f"🔮 Prédiction: {iris.target_names[prediction[0]]}")
print(f"📊 Probabilités: {proba[0]}")