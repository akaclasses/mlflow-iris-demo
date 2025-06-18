"""
Démonstration MLflow Projects
"""
import subprocess
import sys
import os

print("🚀 Démonstration MLflow Projects")

print("\n1️⃣ Lancement avec paramètres par défaut...")

# Lancer directement le script avec paramètres par défaut (= train_simple.py)
cmd1 = [sys.executable, "src/projects/train_project.py", "--n_estimators", "15", "--max_depth", "3"]
result1 = subprocess.run(cmd1, capture_output=True, text=True)

if result1.returncode == 0:
    print("✅ Expérience 1 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 1: {result1.stderr}")

print("\n2️⃣ Lancement avec paramètres personnalisés...")
print("   Équivalent à: mlflow run . -P n_estimators=20 -P max_depth=5")

# Lancer avec paramètres différents
cmd2 = [sys.executable, "src/projects/train_project.py", "--n_estimators", "20", "--max_depth", "5"]
result2 = subprocess.run(cmd2, capture_output=True, text=True)

if result2.returncode == 0:
    print("✅ Expérience 2 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 2: {result2.stderr}")

print("\n3️⃣ Test avec d'autres paramètres...")
print("   n_estimators=50, max_depth=10")

cmd3 = [sys.executable, "src/projects/train_project.py", "--n_estimators", "50", "--max_depth", "10"]
result3 = subprocess.run(cmd3, capture_output=True, text=True)

if result3.returncode == 0:
    print("✅ Expérience 3 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 3: {result3.stderr}")
