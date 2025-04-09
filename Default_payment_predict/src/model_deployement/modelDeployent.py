import pickle
import pandas as pd
import os
##
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
##
from dotenv import load_dotenv
load_dotenv()


ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

# Initialize Arize client with your space key and api key
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Define the schema for your data
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["credit_lines_outstanding", "loan_amt_outstanding",
                          "total_debt_outstanding", "income",
                          "years_employed", "fico_score"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)


def get_model(models):
    return pickle.load(open(models, "rb"))


def model_pred(features, models):
    model = get_model(models)
    prediction = model.predict([features])
    return prediction


def model_arize(features, model, prediction, actual_label):
    # Log the prediction to Arize
    timestamp = pd.Timestamp.now()

    # Log the prediction to Arize
    data = {
        # Unique ID for each prediction
        "prediction_id": [str(timestamp.timestamp())],
        "timestamp": [timestamp],
        "credit_lines_outstanding": [features[0]],
        "loan_amt_outstanding": [features[1]],
        "total_debt_outstanding": [features[2]],
        "income": [features[3]],
        "years_employed": [features[4]],
        "fico_score": [features[5]],
        "prediction_label": [int(prediction)],
        "actual_label": [actual_label]
    }
    dataframe = pd.DataFrame(data)

    try:
        response = arize_client.log(
            dataframe=dataframe,
            model_id=model,
            model_version="v1",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
            # features=features,
            # prediction_label = [int(prediction[0])],
            schema=schema
        )

        if response.status_code != 200:
            print(f"Failed to log data to Arize: {response.text}")
        else:
            print("Successfully logged data to Arize")
    except Exception as e:
        print(f"An error occured: {e}")
