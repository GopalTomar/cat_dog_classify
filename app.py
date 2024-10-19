import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
import gdown

# Replace this with your actual Google Drive link for the .h5 file
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1-zNCcGUSTA4kC6vtl9KliiIBP459O_2n/view?usp=drive_link"

# Function to download the model from Google Drive
def download_model():
    try:
        # Using gdown to download the model file from Google Drive
        gdown.download(GOOGLE_DRIVE_LINK, 'cats_vs_dogs_model.h5', quiet=False)
        model = load_model('cats_vs_dogs_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load your trained model
model = download_model()

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Resize to model input size
        img_array = image.img_to_array(img) / 255.0  # Scale pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Define directories for the cat and dog images
cat_directory = r"C:\Users\Acer\Downloads\dog_cat_project\test_set\cats"
dog_directory = r"C:\Users\Acer\Downloads\dog_cat_project\test_set\dogs"

# Create pages for the Streamlit app
PAGES = {
    "Home": "home",
    "Image Classification": "classification",
}

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", list(PAGES.keys()))

    # Homepage with background image
    if page == "Home":
        st.title("Welcome to Cat vs Dog Image Classification")
        st.image(r"C:\Users\Acer\Downloads\dog_cat_project\WhatsApp Image 2024-10-17 at 08.39.39_0305c39d.jpg", use_column_width=True)

    # Classification Page
    elif page == "Image Classification":
        st.title("Cat vs Dog Image Classification")

        # File uploader for manual upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key='upload')

        # Session state to track whether an image is uploaded
        if uploaded_file is not None:
            st.session_state.uploaded = True
        else:
            st.session_state.uploaded = False

        # Dropdown for folder selection (disabled if an image is uploaded)
        folder_selection = st.selectbox("Select folder:", ["Cats", "Dogs"], disabled=st.session_state.uploaded)

        # Get images based on folder selection (only if not uploading)
        if not st.session_state.uploaded:
            if folder_selection == "Cats":
                image_list = [f for f in os.listdir(cat_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            else:
                image_list = [f for f in os.listdir(dog_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Dropdown selection for images from the selected folder
            image_selection = st.selectbox("Select an image from the folder:", image_list)

        # Process uploaded file
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                # Preprocess the uploaded image
                img_array = load_and_preprocess_image(uploaded_file)

                if img_array is not None:  # Ensure img_array is not None before prediction
                    # Make prediction
                    prediction = model.predict(img_array)
                    predicted_class = 'Dog' if prediction[0][0] >= 0.5 else 'Cat'  # Assuming sigmoid activation for binary classification

                    # Display the result
                    st.write(f"Prediction: {predicted_class}")
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")

        # Process selected image from dropdown (only if no uploaded image)
        if not st.session_state.uploaded and image_selection:
            try:
                # Determine the full path of the selected image based on folder selection
                if folder_selection == "Cats":
                    img_path = os.path.join(cat_directory, image_selection)
                else:
                    img_path = os.path.join(dog_directory, image_selection)

                # Display the selected image
                st.image(img_path, caption="Selected Image from Folder", use_column_width=True)

                # Preprocess the selected image
                img_array = load_and_preprocess_image(img_path)

                if img_array is not None:  # Ensure img_array is not None before prediction
                    # Make prediction
                    prediction = model.predict(img_array)
                    predicted_class = 'Dog' if prediction[0][0] >= 0.5 else 'Cat'  # Assuming sigmoid activation for binary classification

                    # Display the result
                    st.write(f"Prediction: {predicted_class}")
            except Exception as e:
                st.error(f"Error processing selected image from folder: {e}")

if __name__ == "__main__":
    main()
