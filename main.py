import streamlit as st
import cv2
import numpy as np
from cnn_model import load_model  # Ensure this is correctly importing your model loading function
from apply_animal_filters import apply_filters  # Ensure this is correctly importing your filter application function

def main():
    st.title("Pick your favorite filter!")

    st.sidebar.title("Controls")
    start_button = st.sidebar.button("Start Camera", key="start_button")
    stop_button = st.sidebar.button("Stop Camera", key="stop_button")

    filter_option = st.sidebar.selectbox(
        "Choose your filter",
        ["bear.png", "cat.png", "dog.png", "output.PNG", "pig.png"],
        key="filter_option"
    )

    # Display two initial empty windows side by side
    stframe_keypoints = st.empty()
    stframe_filter = st.empty()
    
    empty_image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Set window size to Full HD
    stframe_keypoints.image(empty_image, channels="BGR", caption="Screen with facial Keypoints predicted", use_column_width=True)
    stframe_filter.image(empty_image, channels="BGR", caption="Screen with filter", use_column_width=True)
    
    if start_button:
        run_camera(stframe_keypoints, stframe_filter, filter_option)
    elif stop_button:
        st.stop()

def run_camera(stframe_keypoints, stframe_filter, filter_option):
    model = load_model("models/CNN")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    camera = cv2.VideoCapture(0)

    while True:
        # Read data from the webcam
        ret, image = camera.read()

        if not ret:
            break

        image_copy = np.copy(image)
        image_copy_1 = np.copy(image)

        # Convert RGB image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Identify faces in the webcam using haar cascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces_keypoints = []

        # Loop through faces
        for x, y, w, h in faces:
            # Crop Faces
            face = gray[y : y + h, x : x + w]

            # Scale Faces to 96x96
            scaled_face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)

            # Normalize images to be between 0 and 1
            input_image = scaled_face / 255.0

            # Format image to be the correct shape for the model
            input_image = np.expand_dims(input_image, axis=0)
            input_image = np.expand_dims(input_image, axis=-1)

            # Use model to predict keypoints on image
            face_points = model.predict(input_image)[0]

            # Adjust keypoints to coordinates of original image
            face_points[0::2] = face_points[0::2] * w / 2 + w / 2 + x
            face_points[1::2] = face_points[1::2] * h / 2 + h / 2 + y
            faces_keypoints.append(face_points)

            # Plot facial keypoints on image
            for point in range(15):
                cv2.circle(
                    image_copy,
                    (int(face_points[2 * point]), int(face_points[2 * point + 1])),
                    2,
                    (255, 255, 0),
                    -1,
                )

        # Apply filter
        filter_image = apply_filters(faces_keypoints, image_copy_1, filter_option)

        # Display images using Streamlit
        stframe_keypoints.image(
            image_copy,
            channels="BGR",
            caption="Screen with facial Keypoints predicted",
        )
        stframe_filter.image(filter_image, channels="BGR", caption="Screen with filter")

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
