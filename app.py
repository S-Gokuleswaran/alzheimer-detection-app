import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import io
import plotly.express as px
import pandas as pd

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model_path = "Alzheimer_CNN2d.h5"
model = tf.keras.models.load_model(model_path)

class_labels = [
    'Mild_Demented',
    'Moderate_Demented',
    'Non_Demented',
    'Very_Mild_Demented'
]

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="Alzheimer Detection System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Alzheimer Disease Detection System")

# -------------------------------------------------------
# SESSION STORAGE (for dashboard)
# -------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------
# TABS
# -------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🧠 Prediction",
    "📊 Analytics Dashboard",
    "📚 About Disease"
])

# -------------------------------------------------------
# HOME TAB
# -------------------------------------------------------

with tab1:

    st.header("Welcome")

    image_url = "https://news.mit.edu/sites/default/files/images/202309/MIT-AlzGenome-01-press.jpg"
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    st.image(img, use_container_width=True)

    st.markdown("""
### AI Powered Alzheimer Detection

This application uses **Deep Learning CNN models** to classify Alzheimer disease stages from **MRI Brain scans**.

### Features

✔ Upload MRI Image  
✔ AI Disease Classification  
✔ Confidence Score  
✔ Risk Level Detection  
✔ Probability Graph  
✔ Prediction Analytics Dashboard  

### Disease Stages

• Non Demented  
• Very Mild Demented  
• Mild Demented  
• Moderate Demented
""")

# -------------------------------------------------------
# PREDICTION TAB
# -------------------------------------------------------

with tab2:

    st.header("MRI Scan Prediction")

    uploaded_image = st.file_uploader(
        "Upload an MRI Brain Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        st.subheader("Uploaded Image")

        st.image(uploaded_image, width=250)

        image = Image.open(uploaded_image)
        image = np.array(image)

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        image = tf.image.resize(image, (128, 128))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0)

        with st.spinner("Analyzing MRI image..."):
            predictions = model.predict(image)

        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(predictions)) * 100

        # store for dashboard
        st.session_state.history.append({
            "Prediction": predicted_label,
            "Confidence": confidence
        })

        # ------------------------------------------------
        # RESULT
        # ------------------------------------------------

        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {predicted_label}")

        st.progress(int(confidence))

        st.info(f"Confidence Score: {confidence:.2f}%")

        # ------------------------------------------------
        # RISK LEVEL
        # ------------------------------------------------

        st.subheader("AI Risk Assessment")

        if predicted_label == "Non_Demented":
            st.success("🟢 Risk Level: LOW")

        elif predicted_label == "Very_Mild_Demented":
            st.warning("🟡 Risk Level: MILD")

        elif predicted_label == "Mild_Demented":
            st.warning("🟠 Risk Level: MODERATE")

        else:
            st.error("🔴 Risk Level: HIGH")

        # ------------------------------------------------
        # PROBABILITY GRAPH
        # ------------------------------------------------

        st.subheader("Prediction Probability")

        probabilities = predictions[0] * 100

        df = pd.DataFrame({
            "Disease Stage": class_labels,
            "Probability": probabilities
        })

        fig = px.bar(
            df,
            x="Disease Stage",
            y="Probability",
            text="Probability",
            color="Disease Stage",
            template="plotly_dark"
        )

        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------
        # DOWNLOAD RESULT
        # ------------------------------------------------

        result_text = f"""
Alzheimer MRI Prediction Report

Predicted Class : {predicted_label}

Confidence Score : {confidence:.2f} %

AI Risk Level Assessment Generated
"""

        st.download_button(
            label="📄 Download Result",
            data=result_text,
            file_name="Alzheimer_Prediction_Result.txt"
        )

# -------------------------------------------------------
# ANALYTICS DASHBOARD
# -------------------------------------------------------

with tab3:

    st.header("Prediction Analytics Dashboard")

    if len(st.session_state.history) == 0:

        st.info("No predictions yet. Upload MRI scans to generate analytics.")

    else:

        df = pd.DataFrame(st.session_state.history)

        st.subheader("Prediction Records")

        st.dataframe(df)

        # total predictions
        st.metric("Total Predictions", len(df))

        # ------------------------------------------------
        # CLASS DISTRIBUTION
        # ------------------------------------------------

        st.subheader("Disease Class Distribution")

        fig = px.pie(
            df,
            names="Prediction",
            title="Alzheimer Stage Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # ------------------------------------------------
        # CONFIDENCE TREND
        # ------------------------------------------------

        st.subheader("Confidence Trend")

        df["Prediction Number"] = range(1, len(df)+1)

        fig2 = px.line(
            df,
            x="Prediction Number",
            y="Confidence",
            markers=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig2)

# -------------------------------------------------------
# ABOUT TAB
# -------------------------------------------------------

with tab4:

    st.header("About Alzheimer Disease")

    st.markdown("""
### What is Alzheimer Disease?

Alzheimer's disease is a **progressive neurodegenerative disorder** affecting memory and thinking ability.

### Global Statistics

• Over **55 million people** live with dementia globally  
• Alzheimer accounts for **60–70% of cases**

### Symptoms

• Memory loss  
• Confusion  
• Difficulty thinking  
• Behavioral changes

### Dataset Source

MRI datasets from:

**Alzheimer's Disease Neuroimaging Initiative (ADNI)**

### Importance of AI

AI can help detect Alzheimer disease **earlier using MRI scans**.
""")

    st.warning("⚠ AI assists doctors and does not replace medical diagnosis.")