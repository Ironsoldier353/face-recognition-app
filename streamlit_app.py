import streamlit as st
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from PIL import Image

# Initialize face detector and recognizer
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Streamlit interface
st.title("Real-Time Face Verification")

# Upload image for reference
uploaded_image = st.file_uploader("Upload a reference image for verification", type=["jpg", "jpeg", "png"])

# Function to extract the face embedding from an image
def extract_embedding(image):
    face = mtcnn(image)
    if face is None:
        return None
    face_embedding = model(face.unsqueeze(0))
    return face_embedding

# Once the user uploads an image
if uploaded_image:
    reference_image = Image.open(uploaded_image)
    st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)
    reference_image_rgb = np.array(reference_image.convert('RGB'))
    reference_embedding = extract_embedding(reference_image_rgb)

    if reference_embedding is None:
        st.error("No face detected in the uploaded reference image.")
    else:
        reference_embedding = reference_embedding.view(-1)

        if st.button("Start Webcam", key="start_webcam"):
            cap = cv2.VideoCapture(0)
            st.text("Webcam is active...")

            # Create a placeholder for the video stream
            video_placeholder = st.empty()
            stop_button = st.button("Stop Webcam", key="stop_webcam")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Webcam not detected.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_embedding = extract_embedding(frame_rgb)

                if current_embedding is not None:
                    current_embedding = current_embedding.view(-1)
                    similarity = 1 - cosine(reference_embedding.detach().numpy(), current_embedding.detach().numpy())

                    if similarity > 0.7:
                        label = f"Verified: {similarity:.2f}"
                        color = (0, 255, 0)
                    else:
                        label = f"Not Verified: {similarity:.2f}"
                        color = (0, 0, 255)

                    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # Break the loop when Stop Webcam button is pressed
                if stop_button:
                    break

            cap.release()
            cv2.destroyAllWindows()
