# This is the file that implements a flask server to do inferences

from __future__ import print_function

import io
from tensorflow.keras.models import load_model
import os
import boto3
from urllib.parse import urlparse
import flask
import pandas as pd
from PIL import Image
import numpy as np

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None

    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    @classmethod
    def extract_bucket_name(cls, s3_path):
        """Extract the bucket name from an S3 path."""
        parsed_url = urlparse(s3_path)
        if parsed_url.scheme == 's3':
            bucket_name = parsed_url.netloc
            return bucket_name
        else:
            raise ValueError("Invalid S3 path. It must start with 's3://'")
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            model_path = os.path.join('/opt/ml/model', 'trained_model.h5')
            cls.model = load_model(model_path)
        return cls.model

    @classmethod
    def read_image_from_s3(cls, bucket_name, key):
        """Read image data from S3."""
        try:
            response = cls.s3_client.get_object(Bucket=bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            print(f"Error reading image from S3: {e}")
            return None

    @classmethod
    def predict(cls, input_data):
        """For the input, do the predictions and return them.

        Args:
            input_data (a pandas DataFrame): The data on which to do the predictions. 
                There will be one prediction per row in the DataFrame."""
        model = cls.get_model()
        predictions = []
        
        for _, row in input_data.iterrows():
            image_arrays = []
            for path in row:
                bucket_name = cls.extract_bucket_name(path)
                key = path.split(bucket_name + "/")[1]
                try:
                    img_data = cls.read_image_from_s3(bucket_name, key)
                    if img_data is not None:
                        img = Image.open(io.BytesIO(img_data))
                        img = img.resize((255, 255))
                        img_array = np.array(img) / 255.0
                        image_arrays.append(img_array)
                except Exception as e:
                    print(f"Error reading image {key}: {e}")
                    break
            if len(image_arrays) == 5:
                data = np.stack(image_arrays, axis=-1)
                data = np.array(data)

                data = np.split(data, 5, axis=-1)
                for i in range(5):
                    data[i] = np.squeeze(data[i], axis=-1)
                predictions.append(model.predict(tuple(data)))
            else:
                print("Skipping prediction for row with missing images.")

        return predictions



# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data"""
    try:
        input_data = flask.request.json
        if input_data is None or "input_data" not in input_data:
            return flask.Response(
                response="No input data found in request", status=400, mimetype="text/plain"
            )
        input_data = input_data["input_data"]
        input_array = np.array(input_data)
        input_df = pd.DataFrame(input_array)

        predictions = ScoringService.predict(input_df)
        predictions_list = predictions.tolist()
        response = {"predictions": predictions_list}

        return flask.jsonify(response), 200
    
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        return flask.jsonify({"error": error_message}), 500