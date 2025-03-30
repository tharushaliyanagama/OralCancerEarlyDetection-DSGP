from flask import Flask, request, render_template, redirect, url_for, jsonify
from pymongo import MongoClient, errors
import re
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb+srv://sithmi:ef76hPUdVohpVAZX@cluster0.s4n7e.mongodb.net/User_Details?retryWrites=true&w=majority')
db = client['User_Details']
collection = db['users']

# Check MongoDB connection
try:
    #client.server_info()
    print("Connected to MongoDB successfully.")
except errors.ServerSelectionTimeoutError as err:
    print("Failed to connect to MongoDB:", err)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dataupload')
def dataupload():
    return render_template('dataupload.html')

@app.route('/imageupload')
def imageupload():
    return render_template('imageupload.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register')
def signup():
    return render_template('signup.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/signup', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm-password']

    if len(username) < 3:
        return render_template('signup.html', message="Username must be at least 3 characters long.")
    
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return render_template('signup.html', message="Invalid email format.")
    
    if len(password) < 6:
        return render_template('signup.html', message="Password must be at least 6 characters long.")
    
    if password != confirm_password:
        return render_template('signup.html', message="Passwords do not match.")
    
    if collection.find_one({'email': email}):
        return render_template('signup.html', message="Email already registered.")
    
    if collection.find_one({'username': username}):
        return render_template('signup.html', message="Username already exists.")
    
    user_data = {'username': username, 'email': email, 'password': password}
    try:
        collection.insert_one(user_data)
        return redirect(url_for('dataupload'))
    except errors.PyMongoError as e:
        print(f"Error inserting data: {e}")
        return render_template('signup.html', message="Error occurred during registration. Please try again.")

@app.route('/login', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']

    try:
        user = collection.find_one({'email': email, 'password': password})
        
        if user:
            return redirect(url_for('dataupload'))
        else:
            return render_template('login.html', message="Invalid email or password. Please try again.")
    except errors.PyMongoError as e:
        print(f"Error finding user: {e}")
        return render_template('login.html', message="An error occurred. Please try again later.")


# Load the saved model and scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
model_features = feature_names[:-1]

with open("models/Random_Forest_Classifier.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Mapping of prediction to risk levels
risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = request.form.get('gender')
        tobacco = request.form.get('tobacco')
        alcohol = request.form.get('alcohol')
        hpv = request.form.get('hpv')
        socioeconomic = request.form.get('socioeconomic')
        age_group = request.form.get('age_group')
        continent = request.form.get('continent')

        print(gender, continent, age_group, socioeconomic, tobacco, alcohol, hpv)

        # Mappings
        gender_map = {'male': 0, 'female': 1}
        tobacco_map = {'yes': 1, 'no': 0}
        alcohol_map = {'yes': 1, 'no': 0}
        hpv_map = {'yes': 1, 'no': 0}
        socioeconomic_map = {'low': 0, 'middle': 1, 'high': 2}
        age_group_map = {'young': 0, 'middle': 1, 'older': 2}

        user_data = {
            'Gender': gender_map.get(gender, 0),
            'Tobacco_Use_Yes': tobacco_map.get(tobacco, 0),
            'Alcohol_Use_Yes': alcohol_map.get(alcohol, 0),
            'HPV_Related_Yes': hpv_map.get(hpv, 0),
            'Socioeconomic_Status_Low': 1 if socioeconomic == "Low" else 0,
            'Age_group': 1 if age_group == "Middle" else 0
        }

        # Convert user data to DataFrame and align with model features
        input_df = pd.DataFrame([user_data])

        # Reindex to match the model features
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Predict the risk using the Random Forest model
        predicted_risk = model_rf.predict(input_scaled)[0]  # Get the predicted value

        # Map the prediction to a risk level
        risk_level = risk_mapping.get(predicted_risk, "Unknown")

        print("Predicted Risk Level:", risk_level)

        return jsonify({"prediction": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)