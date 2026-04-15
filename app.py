import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coral Health AI", page_icon="🪸", layout="centered")

# --- STYLE ---
# Using unsafe_allow_html to enable custom CSS for the dashboard look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #007bff; 
        color: white; 
        font-weight: bold;
    }
    .result-box { 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        margin-top: 20px; 
    }
    .bleached { 
        background-color: #ffebee; 
        border: 2px solid #ff1744; 
        color: #b71c1c; 
    }
    .healthy { 
        background-color: #e8f5e9; 
        border: 2px solid #00c853; 
        color: #1b5e20; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Must match the architecture used in your training script
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load your improved weights
    try:
        model.load_state_dict(torch.load('best_resnet18_coral.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ 'resnet18_coral.pth' not found. Please ensure the file is in the same directory.")
        return None, device

# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- APP INTERFACE ---
st.title("🪸 Coral Bleaching Detector")
st.write("Upload an underwater photo to analyze coral health using your fine-tuned ResNet-18 model.")

model, device = load_resnet()

if model:
    uploaded_file = st.file_uploader("Choose a coral image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Diagnostic"):
            with st.spinner('AI is analyzing textures and color patterns...'):
                # Process image
                img_tensor = preprocess_image(image).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                
                # Results mapping
                class_names = ['Bleached', 'Healthy']
                result = class_names[prediction.item()]
                score = confidence.item() * 100
                
                # Dynamic visual feedback
                style_class = "bleached" if result == "Bleached" else "healthy"
                icon = "⚠️" if result == "Bleached" else "✅"
                
                # Display Results
                st.markdown(f"""
                    <div class="result-box {style_class}">
                        <h1 style='margin:0;'>{icon} {result}</h1>
                        <p style='font-size: 1.2em;'>Confidence: <strong>{score:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Expert Recommendations
                if result == "Bleached":
                    st.error("**Finding:** This specimen shows signs of thermal stress and loss of symbiotic algae.")
                else:
                    st.success("**Finding:** This specimen appears to have healthy pigment density.")

# --- FOOTER ---
st.divider()
st.caption("ResNet-18 Deep Learning Classifier | Research Tool for Marine Biology")