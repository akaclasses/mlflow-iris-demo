"""
Script d'entraînement paramétrable pour MLflow Projects
"""
import argparse
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Parser les arguments (valeurs par défaut = train_simple.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=15)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()
    
    mlflow.set_experiment("iris-classification-project")
    with mlflow.start_run():

        # Logger les paramètres
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        # Charger et diviser les données
        iris = datasets.load_iris()
        X, y = iris.data, iris.target        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Entraîner le modèle
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Évaluer et logger
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        
        # Sauvegarder le modèle
        mlflow.sklearn.log_model(model, "iris_model")
        
        print(f"✅ Modèle entraîné - Précision: {accuracy:.2%}")

if __name__ == "__main__":
    main()