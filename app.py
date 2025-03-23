from flask import Flask, request, render_template, redirect, url_for
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
    client.server_info()
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

with open("models/Random_Forest_Classifier.pkl", "rb") as f:
    model_rf = pickle.load(f)

risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    tobacco = request.form['tobacco']
    alcohol = request.form['alcohol']
    hpv = request.form['hpv']
    socioeconomic = request.form['socioeconomic']
    age_group = request.form['age_group']
    
    gender_map = {"Male": 0, "Female": 1}
    yes_no_map = {"Yes": 1, "No": 0}
    socioeconomic_map = {"Low": 0, "Middle": 1, "High": 2}
    age_group_map = {"Young": 0, "Middle": 1, "Older": 2}

    input_data = {
        "Gender_Male": gender_map[gender],
        "Tobacco_Use_Yes": yes_no_map[tobacco],
        "Alcohol_Use_Yes": yes_no_map[alcohol],
        "HPV_Related_Yes": yes_no_map[hpv],
        "Socioeconomic_Status_Low": 1 if socioeconomic == "Low" else 0,
        "Age_group_Middle": 1 if age_group == "Middle" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names[:-1], fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model_rf.predict(input_scaled)[0]
    predicted_risk = risk_mapping.get(prediction, "Unknown")

    return render_template('dataupload.html', prediction=predicted_risk)

if __name__ == '__main__':
    app.run(debug=True)
