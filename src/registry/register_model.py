"""
Enregistrer un modèle dans le Registry MLflow
"""
import mlflow

# Nom du modèle dans le registry
model_name = "iris-classifier"

print("1️⃣ Recherche du meilleur modèle...")

# Trouver la meilleure expérience
experiment = mlflow.get_experiment_by_name("iris-classification")
if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if not runs.empty:
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        accuracy = best_run['metrics.accuracy']
        
        print(f"✅ Meilleure expérience trouvée (accuracy: {accuracy:.2%})")
        
        print("\n2️⃣ Enregistrement dans le Registry...")
        
        # Enregistrer le modèle
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Modèle enregistré:")
        print(f"   - Nom: {model_name}")
        print(f"   - Version: {result.version}")
        print(f"🔍 Voir le registry: mlflow ui")
    else:
        print("❌ Aucune expérience trouvée. Lancez d'abord train_with_tracking.py")
else:
    print("❌ Expérience 'iris-classification' non trouvée. Lancez d'abord train_with_tracking.py")