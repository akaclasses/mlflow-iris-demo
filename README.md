# MLflow Iris Demo 🌸

Démonstration des 4 composants MLflow avec classification d'iris.  
🎯 **Objectif pédagogique** : Comprendre MLflow en 30 minutes.

## 🚀 Quickstart

```bash
# Installation
pip install mlflow scikit-learn

# Test des 4 composants (dans l'ordre)
python src/train_simple.py                    # 1. Baseline sans MLflow
python src/tracking/train_with_tracking.py    # 2. 📈 Tracking
python src/save_load_model.py                 # 3. 💾 Models  
python src/load_model.py                      # 3b 💾 Models  
python src/registry/register_model.py         # 4. 📚 Registry
python src/run_project.py                     # 5. 🚀 Projects

# Interface graphique
mlflow ui
```

## 📂 Structure

```
mlflow-iris-demo/
├── src/
│   ├── train_simple.py              # Training sans MLflow
│   ├── save_load_model.py           # Models demo
│   ├── run_project.py               # Projects demo
│   ├── tracking/
│   │   └── train_with_tracking.py   # Tracking demo
│   ├── projects/
│   │   └── train_project.py         # Script paramétrable
│   └── registry/
│       └── register_model.py        # Registry demo
├── MLproject                        # Configuration projet
└── requirements.txt                 # Dépendances
```

## 🎓 Apprentissage Progressif

| Script | Composant | Concept | Temps |
|--------|-----------|---------|-------|
| `train_simple.py` | - | Baseline ML classique | 2 min |
| `train_with_tracking.py` | 📈 **Tracking** | Enregistrer expériences | 5 min |
| `save_load_model.py` | 💾 **Models** | Sauvegarder/charger modèles | 5 min |
| `register_model.py` | 📚 **Registry** | Organiser versions | 5 min |
| `run_project.py` | 🚀 **Projects** | Standardiser exécution | 5 min |

## 📊 Interface MLflow UI

Après avoir lancé des scripts, explorez vos résultats :

```bash
mlflow ui
# ➜ http://localhost:5000
```
