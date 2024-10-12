import streamlit as st
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from PIL import Image

# Initialize face detector and recognizer
mtcnn = MTCNN(keep_all=False)  # MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()  # Pre-trained FaceNet model for embedding extraction

# Streamlit interface
st.title("Real-Time Face Verification")

# Upload image for reference
uploaded_image = st.file_uploader("Upload a reference image for verification", type=["jpg", "jpeg", "png"])

# Function to extract the face embedding from an image
def extract_embedding(image):
    # Detect and align the face
    face = mtcnn(image)
    if face is None:
        return None
    # Extract face embedding
    face_embedding = model(face.unsqueeze(0))  # Add batch dimension
    return face_embedding

# Once the user uploads an image
if uploaded_image:
    # Load and display the reference image
    reference_image = Image.open(uploaded_image)
    st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)

    # Convert the uploaded image to RGB format (as MTCNN expects RGB)
    reference_image_rgb = np.array(reference_image.convert('RGB'))

    # Extract the reference embedding
    reference_embedding = extract_embedding(reference_image_rgb)

    if reference_embedding is None:
        st.error("No face detected in the uploaded reference image.")
    else:
        # Flatten the reference embedding to 1D
        reference_embedding = reference_embedding.view(-1)

        # Start webcam for real-time face verification
        cap = cv2.VideoCapture(0)
        
        st.text("Starting webcam for real-time face verification... Press 'q' to stop.")

        # Create a placeholder for the video stream
        video_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not detected.")
                break

            # Convert the webcam frame from BGR to RGB (as MTCNN expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and extract embedding for the current frame
            current_embedding = extract_embedding(frame_rgb)

            if current_embedding is not None:
                # Flatten the current embedding to 1D
                current_embedding = current_embedding.view(-1)

                # Calculate the cosine similarity between the reference and current embedding
                similarity = 1 - cosine(reference_embedding.detach().numpy(), current_embedding.detach().numpy())

                # Set a threshold for similarity (higher values mean closer match)
                if similarity > 0.7:
                    label = f"Verified: {similarity:.2f}"
                    color = (0, 255, 0)  # Green for a positive match
                else:
                    label = f"Not Verified: {similarity:.2f}"
                    color = (0, 0, 255)  # Red for a mismatch

                # Display the result on the frame
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Show the video feed in Streamlit
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
