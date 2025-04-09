import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 1. Chargement des données (tu peux adapter ce chemin si besoin)
df = pd.read_csv("data/data_preped.csv")
X = df.drop(columns=["default"])
y = df["default"]

# 2. Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Définir l’experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan_Predict_Logistic")

# 4. Commencer un run MLflow
with mlflow.start_run(run_name="LogisticRegression_baseline"):
    # Paramètres du modèle
    params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 100,
        "random_state": 42
    }

    # Entraînement
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Évaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Log des params et métriques
    mlflow.log_params(params)
    mlflow.log_metrics({
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_score_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_score_1": report["1"]["f1-score"],
        "roc_auc": roc_auc
    })

    # Log du modèle
    mlflow.sklearn.log_model(model, "model")

    print("✅ Modèle loggué dans MLflow avec succès")

print("✅ SCRIPT TERMINÉ")
