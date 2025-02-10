import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Load the trained model
MODEL_PATH = "NEW_CODE_V5.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Labels for exercises
exercise_labels = [
    "ARMPIT LEFT", "ARMPIT RIGHT",
    "CIRCLE LEFT", "CIRCLE RIGHT",
    "CB LEFT", "CB RIGHT",
    "PENDULUM LEFT", "PENDULUM RIGHT",
    "FLEXION LEFT", "FLEXION RIGHT"
]

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Frozen Shoulder Rehabilitation Model")


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lm_list = []
        self.label = "Warmup..."
        self.n_time_steps = 60
        self.step_size = 45

    def find_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return img, results

    def normalize_landmarks(self, landmarks):
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        normalized_lm = []

        for lm_id in [
            mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
        ]:
            lm = landmarks[lm_id]
            normalized_x = lm.x - right_shoulder.x
            normalized_y = lm.y - right_shoulder.y
            normalized_z = lm.z - right_shoulder.z
            normalized_lm.extend([normalized_x, normalized_y, normalized_z])

        return normalized_lm

    def detect_movement(self, lm_list):
        lm_list = np.expand_dims(np.array(lm_list), axis=0)
        results = model.predict(lm_list)
        self.label = exercise_labels[np.argmax(results)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, results = self.find_pose(img)

        if results.pose_landmarks:
            c_lm = self.normalize_landmarks(results.pose_landmarks.landmark)
            if c_lm:
                self.lm_list.append(c_lm)
                if len(self.lm_list) == self.n_time_steps:
                    self.detect_movement(self.lm_list)
                    self.lm_list = self.lm_list[self.step_size:]

        # Display Exercise Name
        cv2.putText(img, self.label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img


# Streamlit WebRTC
webrtc_streamer(key="frozen-shoulder", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor)
