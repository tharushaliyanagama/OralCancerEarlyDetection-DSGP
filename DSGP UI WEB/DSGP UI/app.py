from flask import Flask, request, render_template, redirect, url_for, jsonify ,session
from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi
import re
import numpy as np
import pandas as pd
import pickle
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms, models
from torchvision import transforms
from PIL import Image
import io
import joblib
import base64
from io import BytesIO
import cv2
import shap
import lime
import lime.lime_tabular
import os
import uuid
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  



app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MongoDB Connection
MONGO_URI = "mongodb+srv://tharusha:tharusharocks123@cluster0.ikcamcj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
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

@app.route('/report')
def report():
    heatmap_filename = session.get("heatmap_filename")
    heatmap_url = url_for('static', filename=f'heatmaps/{heatmap_filename}') if heatmap_filename else None

    return render_template("report.html",
        ml_prediction=session.get("ml_prediction"),
        ml_confidence=session.get("ml_confidence"),
        cnn_prediction=session.get("cnn_prediction"),
        cnn_confidence=session.get("cnn_confidence"),
        multimodal_prediction=session.get("multimodal_prediction"),
        instruction_report=session.get("instruction_report"),
        heatmap_image=heatmap_url,
        explanation=session.get("explanation"),
        shap_explanation=session.get("shap_explanation"),
        lime_explanation=session.get("lime_explanation"),
        shap_image=session.get("shap_image"),
        lime_image=session.get("lime_image")
    )  
        
    # === Instruction Generator (already defined) ===
def generate_instruction_report(user_input: dict) -> str:
    instructions = []
    if user_input["Gender"] == 0:
        instructions.append("Gender (Male): Males statistically show higher rates of oral cancer.")
    else:
        instructions.append("Gender (Female): Lower risk but stay proactive.")
    age_group = user_input["Age_group"]
    if age_group == 2:
        instructions.append("Age: Older age increases vulnerability.")
    elif age_group == 1:
        instructions.append("Age: Prevention at middle age is key.")
    else:
        instructions.append("Age: Build healthy habits early.")
    if user_input.get("Socioeconomic_Status_Low"):
        instructions.append("Socioeconomic: Consider public screening options.")
    elif user_input.get("Socioeconomic_Status_Middle"):
        instructions.append("Socioeconomic: Maintain regular checkups.")
    else:
        instructions.append("Socioeconomic: Utilize quality healthcare access.")
    if user_input.get("Tobacco_Use_Yes"):
        instructions.append("Tobacco: Strongly consider quitting.")
    else:
        instructions.append("Tobacco: Good that you avoid it.")
    if user_input.get("Alcohol_Use_Yes"):
        instructions.append("Alcohol: Try reducing or quitting.")
    else:
        instructions.append("Tobacco: Good that you avoid it.")
    if user_input.get("Alcohol_Use_Yes"):
        instructions.append("Alcohol: Try reducing or quitting.")
    else:
        instructions.append("Alcohol: Good that you avoid it.")
    if user_input.get("HPV_Related_Yes"):
        instructions.append("HPV: Regular follow-up recommended.")
    else:
        instructions.append("HPV: Stay safe and consider vaccination.")
    return "\n".join(instructions)


def generate_heatmap(image_pil, model, target_class=0):
    model.eval()
    
    # Original image dimensions
    orig_width, orig_height = image_pil.size
    
    # Transform for model input (resize to 224x224 as expected by ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0)
    
    # Hook storage
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Register hooks on last convolutional layer
    last_conv_layer = model.layer4[-1].conv2
    last_conv_layer.register_forward_hook(forward_hook)
    last_conv_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    class_idx = target_class if target_class is not None else output.argmax().item()
    score = output[0, class_idx]

    # Backward pass
    model.zero_grad()
    score.backward()

    if gradients is None or activations is None:
        return None, None  # Error handling if hooks fail

    # Compute Grad-CAM
    gradients = gradients.detach().numpy()[0]
    activations = activations.detach().numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)

    # Normalize heatmap
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)

    # Resize heatmap to original image size
    heatmap = cv2.resize(cam, (orig_width, orig_height))
    
    # Analyze heatmap to find the region of highest intensity
    max_intensity = np.max(heatmap)
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)  # (y, x) coordinates of max intensity
    max_y, max_x = max_idx
    height, width = heatmap.shape

    # Determine the region of the image (e.g., top-left, center, bottom-right)
    region = "unknown"
    if max_y < height / 3:
        if max_x < width / 3:
            region = "top-left (possibly upper gums or teeth)"
        elif max_x < 2 * width / 3:
            region = "top-center (possibly upper palate)"
        else:
            region = "top-right (possibly upper gums or teeth)"
    elif max_y < 2 * height / 3:
        if max_x < width / 3:
            region = "center-left (possibly inner cheek)"
        elif max_x < 2 * width / 3:
            region = "center (possibly tongue)"
        else:
            region = "center-right (possibly inner cheek)"
    else:
        if max_x < width / 3:
            region = "bottom-left (possibly lower gums or teeth)"
        elif max_x < 2 * width / 3:
            region = "bottom-center (possibly lower tongue or floor of mouth)"
        else:
            region = "bottom-right (possibly lower gums or teeth)"

    # Create heatmap metadata
    heatmap_metadata = {
        "max_intensity": float(max_intensity),
        "region": region,
        "intensity_description": "high" if max_intensity > 0.7 else "moderate" if max_intensity > 0.3 else "low",
        "max_x": int(max_x),
        "max_y": int(max_y)
    }

    # Generate heatmap image
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    # Adjust overlay opacity to make the original image more visible
    original_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_image, 0.7, heatmap_img, 0.3, 0)  # Increased original image weight to 0.7

    # Convert heatmap to Base64
    _, buffer = cv2.imencode('.png', overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return heatmap_base64, heatmap_metadata

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert to PIL image
    image_pil = Image.open(file.stream).convert("RGB")

    # Generate heatmap
    heatmap_base64 = generate_heatmap(image_pil, resnet50_model)

    return jsonify({"heatmap": heatmap_base64})

def generate_explanation(pred_class: str, confidence: float, heatmap_metadata: dict, image_pil) -> str:
    """Generate a dynamic textual explanation for the CNN prediction using heatmap metadata and image features."""
    # Base explanation with prediction and confidence
    base_explanation = f"The model classified this image as {pred_class} with {confidence:.1f}% confidence.\n"

    # Class-specific context
    class_context = {
        'cancer': "This classification suggests the presence of a potentially cancerous lesion.",
        'non-cancer': "This classification indicates healthy oral tissue with no immediate concerns.",
        'leukoplakia': "This classification suggests a precancerous condition that may require monitoring."
    }
    class_explanation = class_context.get(pred_class, "No specific context available.") + "\n"

    # Confidence context
    confidence_context = ""
    if confidence < 60:
        confidence_context = (
            f"The confidence score is relatively low ({confidence:.1f}%), indicating some uncertainty. "
            "It’s recommended to consult a healthcare professional for a more accurate diagnosis.\n"
        )
    elif confidence < 80:
        confidence_context = (
            f"The confidence score ({confidence:.1f}%) suggests moderate certainty. "
            "Consider a follow-up with a healthcare professional for confirmation.\n"
        )
    else:
        confidence_context = (
            f"The confidence score ({confidence:.1f}%) indicates high certainty in the prediction. "
            "A consultation with a healthcare professional is still advised for confirmation.\n"
        )

    # Heatmap-based explanation
    if heatmap_metadata:
        region = heatmap_metadata.get("region", "unknown region")
        intensity = heatmap_metadata.get("intensity_description", "unknown")
        max_x = heatmap_metadata.get("max_x", 0)
        max_y = heatmap_metadata.get("max_y", 0)

        # Extract features from the highlighted region
        image_np = np.array(image_pil)
        region_size = 20  # Define a small region around the max intensity point
        x_start = max(0, max_x - region_size)
        x_end = min(image_np.shape[1], max_x + region_size)
        y_start = max(0, max_y - region_size)
        y_end = min(image_np.shape[0], max_y + region_size)
        region_patch = image_np[y_start:y_end, x_start:x_end]

        # Analyze color (average RGB values)
        if region_patch.size > 0:
            avg_color = np.mean(region_patch, axis=(0, 1))
            color_description = "mixed colors (possibly variegated)" if np.std(region_patch, axis=(0, 1)).mean() > 30 else "uniform color"
        else:
            color_description = "unknown color"

        # Analyze texture (simple variance-based approach)
        texture_description = "irregular texture" if np.var(region_patch) > 1000 else "smooth texture"

        heatmap_explanation = (
            f"The model focused on a {intensity}-intensity region in the {region}. "
            f"This area shows {color_description} and {texture_description}, which may indicate a suspicious lesion or abnormality "
            f"contributing to the {pred_class} prediction.\n"
        )
    else:
        heatmap_explanation = "No specific region of interest was identified by the model.\n"

    # Combine all parts
    return base_explanation + class_explanation + confidence_context + heatmap_explanation
   
@app.route('/imageupload')
def imageupload():
    return render_template('imageupload.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register')
def signup():
    return render_template('signup.html')

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
model_features = feature_names    

with open("models/Random_Forest_Classifier.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Mapping of prediction to risk levels
risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}    
    
# Load Image Validator Model
image_validator = models.resnet50(weights=None)
image_validator.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(image_validator.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
image_validator.load_state_dict(torch.load("models/image_validator.pt", map_location=torch.device('cpu')))
image_validator.eval()

# Load ResNet50 Image Classification Model
resnet50_model = models.resnet50(weights=None)
resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 3)
resnet50_model.load_state_dict(torch.load("models/resnet50_model.pth", map_location=torch.device('cpu')))
resnet50_model.eval()

# Image Classes
IMAGE_CLASSES = ["cancer", "non-cancer", "leukoplakia"]
   
meta_model = joblib.load("models/Model_Fusiononcooral_meta_model.pkl")
label_encoder = joblib.load("models/Model_Fusiononcooral_label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse Form Inputs
        gender = request.form.get('gender', 'Male').lower()
        age_group = request.form.get('age_group', 'Middle').capitalize()
        socioeconomic = request.form.get('socioeconomic', 'Middle').capitalize()
        tobacco = request.form.get('tobacco', 'No').lower()
        alcohol = request.form.get('alcohol', 'No').lower()
        hpv = request.form.get('hpv', 'No').lower()

        gender_map = {'male': 0, 'female': 1}
        age_map = {'Young': 0, 'Middle': 1, 'Older': 2}

        user_data = {
            'Gender': gender_map.get(gender, 0),
            'Age_group': age_map.get(age_group, 1),
            'Socioeconomic_Status_Low': 1 if socioeconomic == 'Low' else 0,
            'Socioeconomic_Status_Middle': 1 if socioeconomic == 'Middle' else 0,
            'Tobacco_Use_Yes': 1 if tobacco == 'yes' else 0,
            'Alcohol_Use_Yes': 1 if alcohol == 'yes' else 0,
            'HPV_Related_Yes': 1 if hpv == 'yes' else 0
        }

        # Build input_df
        input_df = pd.DataFrame([user_data])
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        # Scale & Predict
        input_scaled = scaler.transform(input_df)
        predicted_class = model_rf.predict(input_scaled)[0]
        probas = model_rf.predict_proba(input_scaled)[0]
        confidence = probas[predicted_class]
        risk_label = risk_mapping.get(predicted_class, "Unknown")
        print("Predicted class:", predicted_class, "Label:", risk_label)  # Debug
        print("Probabilities:", probas)  # Debug

        # SHAP Explanation
        explainer = shap.TreeExplainer(model_rf)
        shap_values = explainer.shap_values(input_scaled)
        shap_values_for_class = shap_values[0, :, predicted_class]
        feature_importance = sorted(
            zip(model_features, shap_values_for_class),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_shap_features = [f"{feat} ({val:.3f})" for feat, val in feature_importance[:3]]

        # Generate SHAP Bar Plot
        try:
            plt.figure(figsize=(10, 6))
            feature_values = input_df.values[0].tolist()
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values_for_class,
                    base_values=explainer.expected_value[predicted_class],
                    data=feature_values,
                    feature_names=input_df.columns.tolist()
                ),
                show=False  # Prevent plt.show() warning
            )
            shap_plot_filename = f"shap_{uuid.uuid4().hex}.png"
            shap_plot_path = os.path.join("static", "plots", shap_plot_filename)
            shap_plot_rel_path = f"plots/{shap_plot_filename}"
            os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=100)
            plt.close()
        except Exception as e:
            print(f"SHAP plot generation failed: {e}")
            shap_plot_rel_path = None

        # LIME Explanation
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, input_df.shape[1])),
            feature_names=input_df.columns.tolist(),
            class_names=["Low Risk", "Medium Risk", "High Risk"],
            mode='classification'
        )
        lime_exp = lime_explainer.explain_instance(
            data_row=input_scaled[0],
            predict_fn=model_rf.predict_proba,
            num_features=len(input_df.columns),
            labels=[predicted_class]
        )
        top_lime_features = lime_exp.as_list(label=predicted_class)[:3]  # Top 3 features

        # Convert to user-friendly text
        lime_text_summary = []
        for feature, value in top_lime_features:  # Unpack the tuple (feature, value)
            friendly_name = {
                "Tobacco_Use_Yes": "Tobacco Use",
                "Alcohol_Use_Yes": "Alcohol Use",
                "HPV_Related_Yes": "HPV Status",
                "Socioeconomic_Status_Middle": "Socioeconomic Status (Middle)",
                "Socioeconomic_Status_Low": "Socioeconomic Status (Low)",
                "Age_group": "Age Group",
                "Gender": "Gender"
            }.get(feature, feature)
            impact = "increases your risk" if value > 0 else "decreases your risk"
            lime_text_summary.append(f"- {friendly_name}: {impact} ({abs(value):.3f})")

        # Join the list into a single string
        lime_text_summary = "\n".join(lime_text_summary)
        session['lime_explanation'] = f"Factors influencing your {risk_label} prediction:\n{lime_text_summary}"

        # Store results in session
        session['ml_prediction'] = risk_label
        session['ml_confidence'] = f"{confidence * 100:.2f}%"
        session['instruction_report'] = generate_instruction_report(user_data)
        session['shap_explanation'] = f"Top SHAP features:\n- " + "\n- ".join(top_shap_features)
        session['shap_image'] = shap_plot_rel_path

        return jsonify({
            "prediction": risk_label,
            "confidence": f"{confidence * 100:.2f}%",
            "shap_explanation": session['shap_explanation'],
            "lime_explanation": session['lime_explanation']
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)})
    
@app.route('/multi_model_predict', methods=['POST'])
def multi_model_predict():
    try:
        data = request.get_json()
        ml_conf_str = data.get('ml_confidence', '0')
        cnn_conf_str = data.get('cnn_confidence', '0')

        ml_conf = float(ml_conf_str.replace('%', '')) / 100.0
        cnn_conf = float(cnn_conf_str.replace('%', '')) / 100.0

        if not (0 <= ml_conf <= 1 and 0 <= cnn_conf <= 1):
            return jsonify({"error": "Confidence values must be between 0% and 100%"}), 400

        multimodal_input = [[ml_conf, cnn_conf]]

        # Predict class label
        multimodal_pred = meta_model.predict(multimodal_input)
        multimodal_label = label_encoder.inverse_transform(multimodal_pred)[0]

        # Predict confidence (max probability)
        if hasattr(meta_model, 'predict_proba'):
            proba = meta_model.predict_proba(multimodal_input)[0]
            confidence_score = float(np.max(proba))  # Highest probability
            confidence_percent = f"{confidence_score * 100:.2f}%"
        else:
            confidence_percent = "N/A"

        return jsonify({
            "prediction": multimodal_label,
            "confidence": confidence_percent
        })

    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}),400


# Ensure the heatmaps directory exists
HEATMAP_DIR = os.path.join("static", "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_pil = Image.open(io.BytesIO(file.read())).convert("RGB")
        file.seek(0)  # Reset file pointer
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join("static/images", filename)
    file.save(image_path)

    # Image Validation
    transform_validate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor_validate = transform_validate(image_pil).unsqueeze(0)

    with torch.no_grad():
        prediction = image_validator(input_tensor_validate)
        is_mouth = (prediction < 0.5).int().item()

    if is_mouth == 0:
        print("Image rejected: not a mouth")
        return jsonify({"result": "Not a mouth image"}), 400

    # ResNet50 Image Processing
    transform_predict = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor_predict = transform_predict(image_pil).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(resnet50_model(input_tensor_predict), dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    # Generate Heatmap and Metadata
    heatmap_base64, heatmap_metadata = generate_heatmap(image_pil, resnet50_model, prediction)

    # Save heatmap to filesystem
    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
    with open(heatmap_path, "wb") as f:
        f.write(base64.b64decode(heatmap_base64))

    # Get Prediction Confidence
    output = resnet50_model(input_tensor_predict)
    confidence, class_idx = torch.max(F.softmax(output, dim=1), 1)
    pred_class = IMAGE_CLASSES[class_idx]

    # Generate Explanation
    explanation = generate_explanation(pred_class, confidence.item() * 100, heatmap_metadata, image_pil)

    # Store results in session
    session['cnn_prediction'] = pred_class
    session['cnn_confidence'] = f"{confidence.item() * 100:.2f}%"
    session['heatmap_filename'] = heatmap_filename
    session['explanation'] = explanation

    return jsonify({
        "result": pred_class,
        "confidence": f"{confidence.item() * 100:.2f}%",
        "explanation": explanation,
        "heatmap_image": heatmap_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
