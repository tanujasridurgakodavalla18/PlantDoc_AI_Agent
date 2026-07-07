import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
import datetime

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("plant_cnn_model.keras")

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

recommendations = {
    "Pepper__bell___Bacterial_spot": "Remove infected leaves, avoid overhead watering, use copper-based fungicide.",
    "Pepper__bell___healthy": "No disease detected. Maintain regular watering.",
    "Potato___Early_blight": "Use fungicide and remove affected leaves.",
    "Potato___Late_blight": "Apply fungicide immediately and destroy infected plants.",
    "Potato___healthy": "Plant is healthy.",
    "Tomato_Bacterial_spot": "Use copper spray and disease-free seeds.",
    "Tomato_Early_blight": "Remove infected leaves and spray fungicide.",
    "Tomato_Late_blight": "Avoid wet leaves and use fungicide.",
    "Tomato_Leaf_Mold": "Improve air circulation.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray neem oil.",
    "Tomato__Target_Spot": "Apply fungicide.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants.",
    "Tomato_healthy": "Plant is healthy."
}

# ==========================
# Prediction
# ==========================
def predict(image):

    if image is None:
        return "Please upload an image.", "", 0

    image = image.convert("RGB")
    image = image.resize((128,128))

    img = np.array(image)/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img, verbose=0)

    index = np.argmax(pred)

    label = class_names[index]
    confidence = float(np.max(pred))*100

    recommendation = recommendations.get(label,"Consult agricultural expert.")

    report=f"""
🌿 Disease : {label}

🎯 Confidence : {confidence:.2f} %

💡 Recommendation :

{recommendation}
"""

    return report,label,confidence


# ==========================
# PDF
# ==========================
def generate_pdf(label,confidence):

    filename="plant_report.pdf"

    c=canvas.Canvas(filename)

    c.setFont("Helvetica-Bold",18)
    c.drawString(80,800,"Plant Disease Report")

    c.setFont("Helvetica",12)
    c.drawString(80,760,f"Disease : {label}")
    c.drawString(80,735,f"Confidence : {confidence:.2f}%")
    c.drawString(80,710,f"Date : {datetime.datetime.now()}")

    c.save()

    return filename


# ==========================
# CSS
# ==========================
css="""
body{
background:#eef7ee;
}

.header{
background:linear-gradient(90deg,#2e7d32,#43a047);
padding:18px;
border-radius:10px;
text-align:center;
font-size:32px;
font-weight:bold;
color:white;
margin-bottom:20px;
}

.subtitle{
text-align:center;
font-size:18px;
margin-bottom:20px;
}

.card{
max-width:650px;
margin:auto;
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 8px 20px rgba(0,0,0,0.15);
}

/* Analyze Button */
.analyze-btn button{
    background:#2E7D32 !important;
    color:white !important;
    border:none !important;
    border-radius:30px !important;
    padding:12px !important;
    font-size:18px !important;
    font-weight:bold !important;
}

.analyze-btn button:hover{
    background:#1B5E20 !important;
}

/* Download Button */
.download-btn button{
    background:#1565C0 !important;
    color:white !important;
    border:none !important;
    border-radius:30px !important;
    padding:12px !important;
    font-size:18px !important;
    font-weight:bold !important;
}

.download-btn button:hover{
    background:#0D47A1 !important;
}
"""

# ==========================
# UI
# ==========================
with gr.Blocks(css=css,title="Plant Disease Detector") as app:

    gr.HTML("<div class='header'>🌿 Plant Disease Detector</div>")

    gr.HTML("<div class='subtitle'>Detect Plant Diseases Instantly using AI</div>")

    with gr.Column(elem_classes="card"):

        img=gr.Image(type="pil",label="Choose Image")

        analyze_btn = gr.Button(
            "🔍 Analyze Image",
            elem_classes="analyze-btn"
        )

        output=gr.Textbox(label="Prediction Report",lines=8)

        label_state=gr.State()

        confidence_state=gr.State()

        analyze_btn.click(
            predict,
            inputs=img,
            outputs=[output,label_state,confidence_state]
        )

        pdf_btn = gr.Button(
            "📄 Download PDF Report",
            elem_classes="download-btn"
        )
        pdf_file=gr.File()

        pdf_btn.click(
            generate_pdf,
            inputs=[label_state,confidence_state],
            outputs=pdf_file
        )

app.launch()