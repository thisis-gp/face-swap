import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import insightface
from io import BytesIO

# Initialize InsightFace model
@st.cache_resource
def initialize_model():
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = initialize_model()

# Function to read image from file
def read_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    return img

# Function to display faces
def display_faces(img, faces, title='Image'):
    fig, axs = plt.subplots(1, len(faces) + 1, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title(title)
    
    for i, face in enumerate(faces):
        bbox = face['bbox']
        bbox = [int(b) for b in bbox]
        face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        axs[i + 1].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        axs[i + 1].axis('off')
        axs[i + 1].set_title(f'Face {i + 1}')
    
    return fig

# Function to swap faces
def face_swap(swapper, target_img, target_faces, source_face):
    res = target_img.copy()
    for face in target_faces:
        res = swapper.get(res, face, source_face)
    return res

# Streamlit app layout
st.title("Live Face Swapping App")
st.write("Upload source and target images to swap faces.")

# File uploader for source image
source_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
# File uploader for target image
target_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if source_file and target_file:
    source_img = read_image(source_file)
    target_img = read_image(target_file)

    # Perform face detection
    source_faces = app.get(source_img)
    target_faces = app.get(target_img)

    st.write(f"Number of faces detected in source image: {len(source_faces)}")
    st.write(f"Number of faces detected in target image: {len(target_faces)}")

    if source_faces and target_faces:
        # Display images and detected faces
        st.pyplot(display_faces(source_img, source_faces, title='Source Image'))
        st.pyplot(display_faces(target_img, target_faces, title='Target Image'))

        swapper = insightface.model_zoo.get_model('./inswapper_128.onnx')

        # Perform face swapping
        res = face_swap(swapper, target_img, target_faces, source_faces[0])

        # Display the result
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption='Target Image with Swapped Faces', use_column_width=True)

        # Provide download option for the result
        result = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', result)
        st.download_button(
            label="Download Result Image",
            data=BytesIO(buffer),
            file_name="swapped_face.jpg",
            mime="image/jpeg"
        )
    else:
        st.write("No faces detected in one or both images. Please upload images with clear faces.")
