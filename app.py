import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import streamlit as st


st.title("X-ray Image Classification using pretrained Xception1 model")
st.header("Covid19, Pnemonia and Normal X-rays Example")
st.text("Upload an X-ray image for classification")


def img_classification(img, trained_model):
    # Load Model
    model = load_model(trained_model)

    # Create an array for the right shape to feed into the model
    data = np.ndarray(shape=(1, 256, 256, 3))
    image = img
    #image sizing
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = image.convert("RGB")


    # Turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)


    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probablity


uploaded_file = st.file_uploader('Upload an Xray...', type = ['jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption ='Uploaded Xray.', use_column_width = True)
    st.write('')
    st.write('Classifying...')
    label = img_classification(image, 'covid_19_xception1.h5')
    if label == 0:
        st.write('Predicted class is Covid19')
    elif label == 1:
        st.write('Predicted class is Normal')
    else:
        st.write('Predicted class is Pneumonia')