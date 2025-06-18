"""
Classification d'iris - Avec MLflow Tracking
Le scientifique organisé qui documente tout ! 🔬📋
"""
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

print("🔬 Expérience ML avec MLflow - Méthode 'scientifique'")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Créer/utiliser une expérience (= projet de recherche)
mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    print("📋 Démarrage d'une nouvelle 'run' (expérience individuelle)")
    
    n_estimators = 15
    max_depth = 3
    test_size = 0.35
    random_seed = 999
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", test_size) 
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("model_type", "RandomForest")
    
    print(f"🧪 Paramètres enregistrés: {n_estimators} arbres, split {test_size}")
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Logger les métadonnées des données
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_classes", len(iris.target_names))
    
    print(f"📊 Données: {len(X)} échantillons, {X.shape[1]} caractéristiques")
    
    # 3. Division train/test (mêmes paramètres)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    # 4. Entraînement
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=random_seed
    )
    model.fit(X_train, y_train)
    
    # 5. Évaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    
    # 6. Sauvegarde automatique et intelligente
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Précision: {accuracy:.2%}")
    print(f"📋 Expérience automatiquement enregistrée dans MLflow !")
