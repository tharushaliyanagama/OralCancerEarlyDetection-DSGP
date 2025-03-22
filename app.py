from flask import Flask, request, render_template, redirect, url_for
from pymongo import MongoClient, errors
import re

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb+srv://sithmi:ef76hPUdVohpVAZX@cluster0.s4n7e.mongodb.net/User_Details?retryWrites=true&w=majority')
db = client['User_Details']
collection = db['users']  # Use the 'users' collection

# Check MongoDB connection
try:
    # Attempt to get server info to confirm connection
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

    # Validation
    if len(username) < 3:
        return render_template('signup.html', message="Username must be at least 3 characters long.")
    
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return render_template('signup.html', message="Invalid email format.")
    
    if len(password) < 6:
        return render_template('signup.html', message="Password must be at least 6 characters long.")
    
    if password != confirm_password:
        return render_template('signup.html', message="Passwords do not match.")
    
    # Check for duplicates
    if collection.find_one({'email': email}):
        return render_template('signup.html', message="Email already registered.")
    
    if collection.find_one({'username': username}):
        return render_template('signup.html', message="Username already exists.")
    
    # Save to MongoDB
    user_data = {
        'username': username,
        'email': email,
        'password': password  # Storing password directly (Not secure)
    }
    
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


if __name__ == '__main__':
    app.run(debug=True)
