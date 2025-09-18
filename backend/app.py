from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/salary/predict', methods=['POST'])
def predict_salary():
    data = request.get_json()
    # Call your ML model here (dummy response for now)
    salary = 50000  
    return jsonify({"predicted_salary": salary})
