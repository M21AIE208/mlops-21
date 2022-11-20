from flask import Flask, request, jsonify
from joblib import load
from utils import find_best_model
import glob
app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    content = request.json
    img1 = content['image1']
    clf_name = content['clf_name']
    if content['clf_name'] != None:
        best_model = load(glob.glob(f"./models/{clf_name}_*.joblib")[0])
    else:
        best_model = load(find_best_model())

    predicted_digit_1 = best_model.predict([img1])
    return jsonify({"predicted_digit_1": str(predicted_digit_1[0])})


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)

