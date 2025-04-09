from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

# Entrainement du modèle logistic
params = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 100,
    "random_state": 888,
}
model_lr = LogisticRegression(**params)
model_lr.fit(X_train_balanced, y_train_balanced)

# Prédiction sur les données de test
y_pred = model_lr.predict(X_test)

# Probabilité d'apparition de la class 1

y_1_prob = model_lr.predict_proba(X_test)[:, 1]

# Metrics Precision, Recall, F1-score

recall_metrics = classification_report(y_test, y_pred, output_dict=True)

roc_auc = roc_auc_score(y_test, y_pred)

metrics = {
           "precision_0": recall_metrics["0"]["precision"],
           "recall_0": recall_metrics["0"]["recall"],
           "f1_score_0": recall_metrics["0"]["f1-score"],
           "precision_1": recall_metrics["1"]["precision"],
           "recall_1": recall_metrics["1"]["recall"],
           "f1_score_1": recall_metrics["1"]["f1-score"],
           "roc_auc": roc_auc
           }


# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="LogisticRegression_newparams", params=params,
                     metrics=metrics, model_name=model_lr,
                     X_val=X_test, artifacts_path="Logistic_predict",
                     experiment_name="Loan_Predict_Logistic"
                      )
mlflow_process.wait()