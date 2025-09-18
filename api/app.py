import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/salary/predict', methods=['POST'])
def predict_salary():
    data = request.get_json() 
    # Ml Model
    salary = 50000
    return jsonify({"predicted_salary": salary})

@app.route('/api/resume_screen/predict', methods=['POST'])
def screen_resume():
    data = request.get_json()
    # Run ML model
    result = {"score": 0.87, "status": "accepted"} 
    return jsonify(result)

# Example: Job fit
@app.route('/api/job_fit/predict', methods=['POST'])
def predict_job_fit():
    data = request.get_json()
    # ML logic
    fit_score = 0.76
    return jsonify({"fit_score": fit_score})


# Example: Candidate priority
@app.route('/api/candidate_priority/predict', methods=['POST'])
def predict_candidate_priority():
    data = request.get_json()
    # Ml model
    priority = "high"
    return jsonify({"priority": priority})

if __name__ == "__main__":
    app.run(debug=True, port=8080, host="127.0.0.1")
