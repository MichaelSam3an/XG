from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load model
model = joblib.load(r"models\xg_model_xgb_20260204_234629.joblib")

# Load preprocess pipeline
preprocess = joblib.load(r"models\xg_preprocess_20260204_234629.joblib")

# Load metadata
with open(r"models\xg_metadata_20260204_234629.json") as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]


@app.route("/")
def home():
    return jsonify({
        "message": "xG API Running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        df = pd.DataFrame([data])

        df = df[feature_cols]

        X = preprocess.transform(df)

        xg = model.predict_proba(X)[0][1]

        return jsonify({
            "xg": round(float(xg), 4)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)
