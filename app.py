import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# Initialize face detector and recognizer
mtcnn = MTCNN(keep_all=False)  # MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()  # Pre-trained FaceNet model for embedding extraction

# Function to extract the face embedding from an image
def extract_embedding(image):
    # Detect and align the face
    face = mtcnn(image)
    if face is None:
        return None
    
    # Extract face embedding
    face_embedding = model(face.unsqueeze(0))  # Add batch dimension
    return face_embedding

# Load reference image using OpenCV
reference_image_path = r'J:\projects\face-recognition-app\profile.jpg'
reference_image = cv2.imread(reference_image_path)

# Check if the image was loaded correctly
if reference_image is None:
    print(f"Error: Could not load image from {reference_image_path}")
    exit()

# Convert the reference image from BGR to RGB (as MTCNN expects RGB)
reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# Extract the reference embedding
reference_embedding = extract_embedding(reference_image_rgb)

# Ensure that the reference embedding is valid
if reference_embedding is None:
    print("No face detected in the reference image.")
    exit()

# Flatten the reference embedding to 1D
reference_embedding = reference_embedding.view(-1)

# Start webcam for real-time face verification
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
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

    # Display the frame with the verification result
    cv2.imshow('Face Verification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
