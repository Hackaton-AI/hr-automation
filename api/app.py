import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

app = Flask(__name__)

SALARY_MODEL_PATH = 'C:\\Users\\Ismael\\Desktop\\projects\\hr-automation\\models\\salary-prediction\\salary_prediction_model.pkl'
SALARY_ENCODER_PATH = 'C:\\Users\\Ismael\\Desktop\\projects\\hr-automation\\models\\salary-prediction\\encoder_columns.pkl'
    
salary_model = joblib.load(SALARY_MODEL_PATH)
encoder_columns = joblib.load(SALARY_ENCODER_PATH)

def predict_salary_function(candidate_info):
    if salary_model is None or encoder_columns is None:
        raise ValueError("Model or encoder not loaded properly")
    
    df = pd.DataFrame([candidate_info])
    
    if 'level' in df.columns:
        level_map = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}
        if df['level'].dtype == 'object':
            df['level'] = df['level'].str.lower().map(level_map)
    
    categorical_cols = ['role', 'degree', 'company_size', 'location']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    for col in encoder_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[encoder_columns]
    
    predicted_salary = salary_model.predict(df_encoded)[0]
    return round(predicted_salary, 2)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/salary/predict', methods=['POST'])
def predict_salary():
    try:
        data = request.get_json()
        
        required_fields = ['role', 'years_experience', 'degree', 'company_size', 'location', 'level']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        result = predict_salary_function(data)
        return jsonify({"predicted_salary": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/resume_screen/predict', methods=['POST'])
def screen_resume():
    data = request.get_json()
    # Run ML model
    result = {"score": 0.87, "status": "accepted"} 
    return jsonify(result)

@app.route('/api/job_fit/predict', methods=['POST'])
def predict_job_fit():
    data = request.get_json()
    # ML logic
    fit_score = 0.76
    return jsonify({"fit_score": fit_score})

@app.route('/api/candidate_priority/predict', methods=['POST'])
def predict_candidate_priority():
    data = request.get_json()
    # Ml model
    priority = "high"
    return jsonify({"priority": priority})


@app.route('/api/salary/test', methods=['GET'])
def test_salary_prediction():
    condidate = request.get_json()
    try:
        # Test with sample candidate
        test_candidate = {
            "role": condidate['role'],
            "years_experience": condidate['years_experience'],
            "degree": condidate['degree'],
            "company_size": condidate['company_size'],
            "location": condidate['location'],
            "level": condidate['level'],
        }
        
        result = predict_salary_function(test_candidate)
        return jsonify({
            "test_candidate": test_candidate,
            "predicted_salary": result,
            "status": "Model working correctly"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "Model test failed"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080, host="127.0.0.1")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# try:
#     if os.path.exists(SALARY_MODEL_PATH) and os.path.exists(SALARY_ENCODER_PATH):
#         salary_model = joblib.load(SALARY_MODEL_PATH)
#         encoder_columns = joblib.load(SALARY_ENCODER_PATH)
#         print("Salary prediction model and encoder loaded successfully")
#     else:
#         salary_model = None
#         encoder_columns = None
#         print(f"Warning: Model files not found")
#         print(f"Model path exists: {os.path.exists(SALARY_MODEL_PATH)}")
#         print(f"Encoder path exists: {os.path.exists(SALARY_ENCODER_PATH)}")
# except Exception as e:
#     salary_model = None
#     encoder_columns = None
#     print(f"Error loading model: {e}")