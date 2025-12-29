# 🍃 Crop Disease Detection App

An AI-powered web application for detecting crop diseases from leaf images using **Convolutional Neural Networks (CNN)**.  
This project focuses on **sustainable agriculture in Bangladesh** by enabling early disease detection.

---

## 🌱 Project Overview

Crop diseases cause significant yield loss every year.  
This application uses a trained deep learning model to identify plant diseases from images, helping farmers and researchers take early action.

---

## 🚀 Features

- 📸 Upload leaf images
- 🤖 CNN-based disease classification
- 📊 Confidence score for predictions
- 🌾 Supports sustainable agriculture
- 🌐 Deployed using Streamlit Cloud

---

## 🧠 Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input Size: 224 × 224 RGB images
- Output: Disease class prediction
- Framework: TensorFlow / Keras

---

## 🗂️ Project Structure

crop-disease-app/ │ ├── app.py               # Main Streamlit application ├── final_model.h5       # Trained CNN model ├── class_names.txt      # Disease class labels ├── requirements.txt     # Required Python packages └── README.md            # Project documentation
---

## ⚙️ Installation (Local Run)

1. Clone the repository:
   ```bash
   git clone https://github.com/zahid397/crop-disease-app.git
   cd crop-disease-app
   Install dependencies:
   pip install -r requirements.txt
   Run the application:
   streamlit run app.py
   Deployment
This app is deployed using Streamlit Cloud.
Steps:
Push code to GitHub
Connect repository in Streamlit Cloud
Select app.py as main file
Deploy 🚀
⚠️ Disclaimer
This system is a prototype AI application.
Predictions are for educational and research purposes only.
For real-world farming decisions, consult agricultural experts.
👨‍💻 Author
Zahid Hasan
AI & Machine Learning Enthusiast
Focused on Applied AI and Automation
🌍 Vision
"Using AI to support farmers and promote sustainable agriculture in Bangladesh."
