import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load model and get expected input shape
model = tf.keras.models.load_model('class8_ker.h5')
input_shapes = [inp.shape for inp in model.inputs]
MAX_SEQ_LENGTH = input_shapes[0][1]
NUM_FEATURES = input_shapes[0][2]

# Load CNN feature extractor
cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

def extract_frame_features(frames):
    frames_resized = np.array([cv2.resize(f, (224, 224)) for f in frames])
    preprocessed = preprocess_input(frames_resized)
    return cnn.predict(preprocessed, verbose=0)

def load_video(path, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)

def predict_video(path):
    frames = load_video(path)
    length = min(len(frames), MAX_SEQ_LENGTH)
    frames = frames[:length]

    features = extract_frame_features(frames)

    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_masks = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features[0, :length, :] = features
    frame_masks[0, :length] = 1

    probs = model.predict([frame_features, frame_masks])[0]
    predicted_index = np.argmax(probs)
    labels = ['PlayingGuitar', 'PlayingTabla', 'Rafting', 'Skiing', 'SkyDiving', 'TableTennisShot', 'Typing', 'YoYo']
    predicted_class = labels[predicted_index]
    return predicted_class, probs

# Streamlit UI
st.title("Video Classifier")
st.write("Upload a short video file for prediction")

video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    st.video(tfile.name)
    
    with st.spinner("Predicting..."):
        label, probs = predict_video(tfile.name)
        st.success(f"**Prediction:** {label}")
        st.write("**Probabilities:**")
        # st.json({ "rafting": float(probs[0]), "skiing": float(probs[1]) })
        st.json({
    "playingguitar": float(probs[0]),
    "playingtabla": float(probs[1]),
    "rafting": float(probs[2]),
    "skiing": float(probs[3]),
    "skydiving": float(probs[4]),
    "tabletennisshot": float(probs[5]),
    "typing": float(probs[6]),
    "yoyo": float(probs[7])
})