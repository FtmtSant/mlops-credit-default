import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/data_preped.csv")



# Séparation du target et des features


def data_preped():

    x = df.drop(["default", "customer_id"], axis=1)
    y = df["default"]

    # Séparation  des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Equilibre des classes

    # Underfiting
    # under_sampler = RandomUnderSampler(sampling_strategy=0.5,
    # random_state=42)
    # X_train_balanced, y_train_balanced = under_sampler.fit_resample(X_train,
    # y_train)

    # overfiting
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test = scaler.transform(X_test)
    return X_train_balanced, X_test, y_train_balanced, y_test

