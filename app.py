import streamlit as st
import tempfile
from PIL import Image, UnidentifiedImageError  # Import UnidentifiedImageError explicitly
from transformers import pipeline


# Load the image classification pipeline from Hugging Face
classifier = pipeline("image-classification")

# Streamlit UI
st.title("Image Classification Web App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    try:
        # Attempt to open the image using Pillow
        with st.spinner('Classifying...'):
            image_pil = Image.open(uploaded_file)
            predictions = classifier(image_pil)

    except UnidentifiedImageError:
        # Handle unsupported image format gracefully
        st.warning("The uploaded image format is not supported. Please try with a JPG, PNG, or JPEG image.")

    except Exception as e:
        # Catch other potential errors
        st.error("An error occurred while processing the image. Please try again.")
        raise e

    # Display predictions
    if predictions:
        st.subheader("Predictions:")
        for pred in predictions:
            st.write(f"Class: {pred['label']}, Probability: {pred['score']:.4f}")
    else:
        st.warning("No predictions were obtained. Please ensure the image is valid and the model is suitable for your task.")
