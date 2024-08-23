import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for
import joblib
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import pickle
from PIL import Image
import json
import numpy as np
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/zidan/OneDrive/Documents/Stuff/crop disease/Web/static/uploads'

# Directories
model_directory = r'C:\Users\zidan\OneDrive\Documents\Stuff\crop disease\Web'  
image_directory = r'C:\Users\zidan\OneDrive\Documents\Stuff\crop disease\Web\static\pics'
chatbot_data_path = r'C:\Users\zidan\OneDrive\Documents\Stuff\crop disease\data\AgroQA Dataset.csv'
plant_disease_model_path = os.path.join(model_directory, 'crop_Disease.keras')

os.chdir(model_directory)

# Crop Recommendation
crop_recommend_model_path = os.path.join(model_directory, 'crop_recommend_model.joblib')
scaler_path = os.path.join(model_directory, 'crop_recommend_scaler.pkl')
crop_recommend_model = joblib.load(crop_recommend_model_path)
scaler = joblib.load(scaler_path)

# Load Crop Production Model
products_rf_model = joblib.load(os.path.join(model_directory, 'Production_random_forest_model.joblib'))
products_label_encoders = joblib.load(os.path.join(model_directory, 'Production_label_encoders.joblib'))

# Load chatbot model and vectorizer
chatbot_model = joblib.load('chatbot_model.joblib')
vectorizer = joblib.load('vectorizer.pkl')
chatbot_data = pd.read_csv(chatbot_data_path)

chatbot_data['Question'] = chatbot_data['Question'].apply(lambda x: x.lower())
chatbot_data['Answer'] = chatbot_data['Answer'].apply(lambda x: x.lower() if isinstance(x, str) else "")

vectorizer = TfidfVectorizer()
chatbot_questions_vectorized = vectorizer.fit_transform(chatbot_data['Question'])

# Load plant disease model
##------NEED KERAS 3.3.3 VERSION TO LOAD MODEL BELOW---------##
plant_disease_model = load_model('C:/Users/zidan/OneDrive/Documents/Stuff/crop disease/Web/plant_disease.keras')
label_encoder_path = 'C:/Users/zidan/OneDrive/Documents/Stuff/crop disease/Web/class_indices.json'

THRESHOLD = 0.8

# Load Json
with open('plants.json', 'r') as treat:
    treatment_info = json.load(treat)

with open('crops.json', 'r') as f:
    crops_info = json.load(f)
    
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        
        nitrogen = float(request.form['N'])
        phosphorous = float(request.form['P'])
        potassium = float(request.form['K'])
        temperature = float(request.form['temperature'])    
        humidity = float(request.form['humidity'])
        pH = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        input_features = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, pH, rainfall]])
        
        input_features_scaled = scaler.transform(input_features)
        
        recommended_crop_index = crop_recommend_model.predict(input_features_scaled)[0]
       
        crop_details = {}
        for crop in crops_info.values():
            for details in crop:
                if details['Name'].lower() == recommended_crop_index.lower():
                    crop_details = details
                    break

        return render_template('crop_result.html', prediction=crop_details)
    
    return render_template('crop_recommendation.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['user_input'].lower()
        
        user_input_vectorized = vectorizer.transform([user_input])
        
        similarities = cosine_similarity(user_input_vectorized, chatbot_questions_vectorized).flatten()
        most_similar_index = np.argmax(similarities)
        max_similarity = similarities[most_similar_index]

        similarity_threshold = 0.2  
        if max_similarity > similarity_threshold:
            question = chatbot_data.loc[most_similar_index, 'Question']
            answer = chatbot_data.loc[most_similar_index, 'Answer']
        else:
            question = user_input
            answer = "I am sorry, I do not have an answer for question. Please try asking another."

        return render_template('chatbot.html', question=question, answer=answer)
    
    return render_template('chatbot.html')

@app.route('/plant_disease', methods=['GET', 'POST'])
def plant_disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            
            predictions = plant_disease_model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_probability = np.max(predictions)

            if predicted_probability < THRESHOLD:
                predicted_class = "Unable To Identify. Try Another Image"
                treatments = None
            else:
                
                predicted_class = list(class_indices.keys())[predicted_index]
                treatments = treatment_info.get(str(predicted_index), [{}])[0]

            return render_template(
                'plant_disease_result.html',
                prediction=predicted_class,
                filename=filename,
                treatments=treatments
            )

    return render_template('plant_disease.html')

@app.route('/products', methods=['GET', 'POST'])
def products():
    if request.method == 'POST':
        year3 = 2024
        item3 = request.form['Item']
        area3 = request.form['Area']
        element3 = 'Production'  # Fixed Element
        
        item_encoded3 = products_label_encoders['Item'].transform([item3])[0]
        area_encoded3 = products_label_encoders['Area'].transform([area3])[0]
        element_encoded3 = products_label_encoders['Element'].transform([element3])[0]

        input_features3 = np.array([[year3, item_encoded3, area_encoded3, element_encoded3]])

        prediction3 = products_rf_model.predict(input_features3)[0]

        return render_template('products_result.html', prediction3=prediction3)

    return render_template('products.html')

if __name__ == '__main__':
    app.run(debug=True)
