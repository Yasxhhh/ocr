import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Path to tesseract.exe
# Update path as per your setup
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Preprocess the image before OCR


def preprocess_image(image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(
        gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise the image using a median filter
    denoised = cv2.medianBlur(binary, 3)

    return denoised

# OCR function to extract text from an image


def ocr_from_image(image, lang='eng', psm=3):
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform OCR using pytesseract
        custom_config = f'--oem 3 --psm {psm}'
        text = pytesseract.image_to_string(
            preprocessed_image, lang=lang, config=custom_config)

        return text
    except Exception as e:
        return str(e)


# Streamlit app
st.title("Image to Text Converter")

# Upload image or capture via camera
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a picture with your camera")

# Display image and extract text
if uploaded_file or camera_image:
    if uploaded_file:
        image = Image.open(uploaded_file)  # Use PIL to open the image
    else:
        image = Image.open(camera_image)

    # Show the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to run OCR
    if st.button("Extract Text"):
        with st.spinner('Processing...'):
            extracted_text = ocr_from_image(image, lang='eng', psm=3)

        st.subheader("Extracted Text:")
        st.write(extracted_text)
