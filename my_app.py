import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -------------------------------
# 🎯 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🩺 Kidney Defect Detector",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🎨 Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: transparent;
        padding: 2rem;
        border-radius: 0;
        box-shadow: none;
        margin-bottom: 1.5rem;
        border: none;
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit default borders */
    [data-testid="stImage"] {
        border: none !important;
    }
    
    .element-container img {
        border: none !important;
    }
    
    /* Remove white backgrounds */
    .uploadedFile {
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        background: transparent !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        opacity: 0.6;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .result-card h2 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .result-card .confidence {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Upload section */
    .uploadedFile {
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        background: transparent !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #333;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🩻 Sidebar Information
# -------------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: white; font-size: 2rem; margin-bottom: 0;'>🧠 AI Diagnostic</h1>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Powered by Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: white; margin-bottom: 0.5rem;'>📊 Classification Categories</h3>
            <ul style='color: rgba(255,255,255,0.9); list-style: none; padding-left: 0;'>
                <li>🟢 <b>Normal</b> - Healthy kidney tissue</li>
                <li>🟡 <b>Cyst</b> - Fluid-filled sacs</li>
                <li>🔵 <b>Stone</b> - Kidney stones detected</li>
                <li>🔴 <b>Tumor</b> - Abnormal growth</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
            <h3 style='color: white; margin-bottom: 0.5rem;'>🔬 Model Specifications</h3>
            <ul style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
                <li><b>Architecture:</b> MobileNet (Fine-tuned)</li>
                <li><b>Input Size:</b> 224x224 pixels</li>
                <li><b>Framework:</b> TensorFlow/Keras</li>
                <li><b>Dataset:</b> CT Kidney Images</li>
                <li><b>Accuracy:</b> Medical-grade precision</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background: rgba(255,100,100,0.2); padding: 1rem; border-radius: 10px; border: 2px solid rgba(255,255,255,0.3);'>
            <h4 style='color: white; margin-bottom: 0.5rem;'>⚠️ Medical Disclaimer</h4>
            <p style='color: rgba(255,255,255,0.9); font-size: 0.85rem; line-height: 1.4;'>
                This application is designed for <b>educational and research purposes only</b>. 
                It is not intended to replace professional medical diagnosis or advice. 
                Always consult qualified healthcare professionals for medical decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🚀 Main App Layout
# -------------------------------
st.markdown("""
    <div class='main-header'>
        <h1>🩺 AI-Powered Kidney Defect Detector</h1>
        <p>Advanced Medical Image Analysis using Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# ⚙️ Load Model
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_prediction_model():
    try:
        model = tf.keras.models.load_model("kidney_mobilenet10.keras")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("Ensure 'kidney_mobilenet11.keras' is in the correct directory and compatible with TensorFlow version.")
        st.stop()

with st.spinner("🔄 Loading AI model..."):
    try:
        model = load_prediction_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# -------------------------------
# 🧩 Helper Functions
# -------------------------------
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
class_colors = ['#FFD700', '#00CC66', '#0099FF', '#FF3333']
class_emojis = ['🟡', '🟢', '🔵', '🔴']

def preprocess_image(image):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"⚠️ Error preprocessing image: {str(e)}")
        return None

def plot_confidence_plotly(probabilities):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=class_names,
        y=probabilities * 100,
        marker=dict(
            color=class_colors,
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "🔍 Prediction Confidence Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333', 'family': 'Arial Black'}
        },
        xaxis_title="Condition",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(size=12, color='#333'),
        height=400,
        margin=dict(t=80, b=60, l=60, r=40),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_recommendation(predicted_class, confidence):
    recommendations = {
        'Normal': {
            'icon': '✅',
            'message': 'No abnormalities detected',
            'action': 'Continue regular health checkups and maintain a healthy lifestyle.',
            'color': 'success-box'
        },
        'Cyst': {
            'icon': '⚠️',
            'message': 'Cyst detected',
            'action': 'Schedule an appointment with a urologist for evaluation. Most cysts are benign but require monitoring.',
            'color': 'warning-box'
        },
        'Stone': {
            'icon': '🔍',
            'message': 'Kidney stone detected',
            'action': 'Consult a urologist immediately. Drink plenty of water and avoid high-oxalate foods.',
            'color': 'warning-box'
        },
        'Tumor': {
            'icon': '🚨',
            'message': 'Tumor detected',
            'action': 'URGENT: Consult an oncologist or urologist as soon as possible for comprehensive evaluation and treatment planning.',
            'color': 'info-box'
        }
    }
    return recommendations.get(predicted_class, recommendations['Normal'])

def generate_pdf_report(image, predicted_class, confidence, predictions, uploaded_filename):
    """Generate a PDF report of the diagnosis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("🩺 Kidney Defect Detector - Medical Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Paragraph(f"<b>Image File:</b> {uploaded_filename}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis Result
    story.append(Paragraph("DIAGNOSIS RESULT", heading_style))
    
    # Result table
    result_data = [
        ['Predicted Condition', predicted_class],
        ['Confidence Level', f'{confidence:.2f}%'],
        ['Analysis Date', report_date]
    ]
    
    result_table = Table(result_data, colWidths=[2.5*inch, 3.5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('BACKGROUND', (1, 0), (-1, -1), colors.HexColor('#f5f7fa')),
        ('TEXTCOLOR', (1, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Detailed Analysis
    story.append(Paragraph("DETAILED CONFIDENCE ANALYSIS", heading_style))
    
    analysis_data = [['Condition', 'Confidence Level']]
    for name, prob, emoji in zip(class_names, predictions[0], class_emojis):
        analysis_data.append([f'{emoji} {name}', f'{prob*100:.2f}%'])
    
    analysis_table = Table(analysis_data, colWidths=[3*inch, 3*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')])
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("MEDICAL RECOMMENDATIONS", heading_style))
    rec = get_recommendation(predicted_class, confidence)
    story.append(Paragraph(f"<b>{rec['icon']} {rec['message']}</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<b>Recommended Action:</b> {rec['action']}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("MEDICAL DISCLAIMER", heading_style))
    disclaimer_text = """This report is generated by an AI-powered diagnostic tool for educational and research 
    purposes only. It is NOT intended to replace professional medical diagnosis, advice, or treatment. 
    The results should be reviewed and verified by qualified healthcare professionals. Always seek the 
    advice of your physician or other qualified health provider with any questions you may have regarding 
    a medical condition."""
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph("© 2025 Kidney Defect Detector | Powered by TensorFlow & Streamlit", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------------
# 📤 Upload Section
# -------------------------------
st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.markdown("### 📤 Upload CT Scan Image")
st.markdown("Please upload a high-quality kidney CT scan in JPEG or PNG format")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    help="Upload a kidney CT scan image for AI analysis"
)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 🩻 Prediction Section
# -------------------------------
if uploaded_file is not None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, width=300, caption="Original CT Scan")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Image Information")
        st.markdown(f"""
        - **Filename:** {uploaded_file.name}
        - **Size:** {uploaded_file.size / 1024:.2f} KB
        - **Dimensions:** {image.size[0]} x {image.size[1]} pixels
        - **Format:** {image.format}
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis Button
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    if st.button("🔬 Analyze Image with AI"):
        with st.spinner("🧠 AI is analyzing your CT scan..."):
            img_array = preprocess_image(image)
            
            if img_array is not None:
                try:
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class = class_names[np.argmax(predictions[0])]
                    confidence = np.max(predictions[0]) * 100
                    class_idx = np.argmax(predictions[0])
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    
                    # Result Card
                    st.markdown(f"""
                        <div class='result-card'>
                            <h2>{class_emojis[class_idx]} Diagnosis Result</h2>
                            <h1 style='font-size: 2.5rem; margin: 1rem 0;'>{predicted_class}</h1>
                            <div class='confidence'>{confidence:.1f}%</div>
                            <p style='font-size: 1.1rem; opacity: 0.9;'>Confidence Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    
                    # Confidence Distribution
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    plot_confidence_plotly(predictions[0])
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    
                    # Detailed Results
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Detailed Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    for i, (name, prob, emoji) in enumerate(zip(class_names, predictions[0], class_emojis)):
                        with [col1, col2, col3, col4][i]:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h3>{emoji}</h3>
                                    <h3>{prob*100:.1f}%</h3>
                                    <p>{name}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    
                    # Recommendations
                    rec = get_recommendation(predicted_class, confidence)
                    st.markdown(f"""
                        <div class='{rec['color']}'>
                            <h3>{rec['icon']} {rec['message']}</h3>
                            <p style='font-size: 1.1rem; margin-top: 0.5rem;'><b>Recommended Action:</b></p>
                            <p style='font-size: 1rem; line-height: 1.6;'>{rec['action']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    
                    # Download Report Section
                    st.markdown("""
                        <div style='text-align: center; padding: 1.5rem 0;'>
                            <h3 style='color: #667eea; margin-bottom: 1rem;'>📄 Download Medical Report</h3>
                            <p style='color: #666;'>Generate a comprehensive PDF report of this diagnosis</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        pdf_buffer = generate_pdf_report(image, predicted_class, confidence, predictions, uploaded_file.name)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"kidney_diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {str(e)}")
else:
    # Empty State
    st.markdown("""
        <div class='custom-card' style='text-align: center; padding: 4rem 2rem;'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>👆 Get Started</h2>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
                Upload a kidney CT scan image to begin AI-powered analysis
            </p>
            <div style='font-size: 4rem; opacity: 0.3;'>🩻</div>
        </div>
    """, unsafe_allow_html=True)

# --------------------------------
# 🧾 Footer
# --------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p style='font-size: 0.9rem;'>
            🧬 Developed with ❤️ using TensorFlow & Streamlit | © 2025 Kidney Defect Detector
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
            Empowering healthcare with artificial intelligence
        </p>
    </div>
""", unsafe_allow_html=True)
