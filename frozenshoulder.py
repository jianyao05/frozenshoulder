import time
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import streamlit as st


class FrozenShoulder:
    def __init__(self, model_path, n_time_steps = 60, step_size = 45):
        # Initialize MediaPipe
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

        # Initialise Model & the thresholds and values
        self.model = tf.keras.models.load_model(model_path)
        self.n_time_steps = n_time_steps
        self.step_size = step_size

        # Labels for Indices of leg landmarks in MediaPipe (leg movement detection)
        self.arm_landmarks = {
            # RIGHT
            'right_shoulder': 12,  # FOR NORMALISATION
            'right_elbow': 14,
            'right_wrist': 16,
            'right_pinky': 18,
            'right_index': 20,
            'right_thumb': 22,
            # LEFT
            'left_shoulder': 11,  # FOR NORMALISATION
            'left_elbow': 13,
            'left_wrist': 15,
            'left_pinky': 17,
            'left_index': 19,
            'left_thumb': 21
        }

        # Labels for exercises
        self.exercise_labels = [
            "ARMPIT LEFT", "ARMPIT RIGHT",
            "CIRCLE LEFT", "CIRCLE RIGHT",
            "CB LEFT", "CB RIGHT",
            "PENDULUM LEFT", "PENDULUM RIGHT",
            "FLEXION LEFT", "FLEXION RIGHT"
        ]

        self.label = "Warmup...."       # types of label for each exercise
        self.lm_list = []  # Store landmark data for prediction

        # Exercise state tracking
        self.list = []
        self.state_sequence = []


        # -------------------------------------------- REPETITION COUNTER -------------------------------------------- #
        self.repetition_left_armpit = 0
        self.repetition_right_armpit = 0
        ###
        self.repetition_left_circle = 0
        self.repetition_right_circle = 0
        ###
        self.repetition_left_cross = 0
        self.repetition_right_cross = 0
        ###
        self.repetition_left_pendulum = 0
        self.repetition_right_pendulum = 0
        ###
        self.repetition_left_flexion = 0
        self.repetition_right_flexion = 0
        ###
        self.repetition_left_towel = 0
        self.repetition_right_towel = 0


        # ------------------------------------------ REPETITION THRESHOLDS ------------------------------------------- #
        ###
        self.repetition_threshold_left_armpit = 0
        self.repetition_threshold_right_armpit = 0
        ###
        self.repetition_threshold_left_circle = 0
        self.repetition_threshold_right_circle = 0
        ###
        self.repetition_threshold_left_cross = 0
        self.repetition_threshold_right_cross = 0
        ###
        self.repetition_threshold_left_pendulum = 0
        self.repetition_threshold_right_pendulum = 0
        ###
        self.repetition_threshold_left_flexion = 0
        self.repetition_threshold_right_flexion = 0
        ###
        self.repetition_threshold_left_towel = 0
        self.repetition_threshold_right_towel = 0

        # ----------------------------------------------- SET COUNTER ------------------------------------------------ #
        ###
        self.set_left_armpit = 0
        self.set_right_armpit = 0
        ###
        self.set_left_circle = 0
        self.set_right_circle = 0
        ###
        self.set_left_cross = 0
        self.set_right_cross = 0
        ###
        self.set_left_pendulum = 0
        self.set_right_pendulum = 0
        ###
        self.set_left_flexion = 0
        self.set_right_flexion = 0
        ###
        self.set_left_towel = 0
        self.set_right_towel = 0


        # ----------------------------------------------- SET THRESHOLDS --------------------------------------------- #
        ###
        self.set_threshold_left_armpit = 0
        self.set_threshold_right_armpit = 0
        ###
        self.set_threshold_left_circle = 0
        self.set_threshold_right_circle = 0
        ###
        self.set_threshold_left_cross = 0
        self.set_threshold_right_cross = 0
        ###
        self.set_threshold_left_pendulum = 0
        self.set_threshold_right_pendulum = 0
        ###
        self.set_threshold_left_flexion = 0
        self.set_threshold_right_flexion = 0
        ###
        self.set_threshold_left_towel = 0
        self.set_threshold_right_towel = 0

        # --------------------------------------------- ANGLE THRESHOLDS --------------------------------------------- #
        ###
        self.angle_left_armpit = 0
        self.angle_right_armpit = 0
        ###
        self.angle_left_circle = 0
        self.angle_right_circle = 0
        ###
        self.angle_left_cross = 0
        self.angle_right_cross = 0
        ###
        self.angle_left_pendulum = 0
        self.angle_right_pendulum = 0
        ###
        self.angle_left_flexion = 0
        self.angle_right_flexion = 0
        ###
        self.angle_left_towel = 0
        self.angle_right_towel = 0
        pass

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    # ------------------- STORES LANDMARK AND COORDINATES IN LIST eg. [id, x, y, z] --------------
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # 480 640
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def normalize_landmarks(self, landmarks):
        right_shoulder_x = landmarks[self.arm_landmarks['right_shoulder']].x
        right_shoulder_y = landmarks[self.arm_landmarks['right_shoulder']].y
        right_shoulder_z = landmarks[self.arm_landmarks['right_shoulder']].z
        left_shoulder_x = landmarks[self.arm_landmarks['left_shoulder']].x
        left_shoulder_y = landmarks[self.arm_landmarks['left_shoulder']].y
        left_shoulder_z = landmarks[self.arm_landmarks['left_shoulder']].z

        normalized_lm = []

        # Normalize right arm landmarks
        normalized_lm.extend([right_shoulder_x, right_shoulder_y, right_shoulder_z])
        for lm_id in ['right_elbow', 'right_wrist', 'right_pinky', 'right_index', 'right_thumb']:
            landmark = landmarks[self.arm_landmarks[lm_id]]
            normalized_x = landmark.x - right_shoulder_x
            normalized_y = landmark.y - right_shoulder_y
            normalized_z = landmark.z - right_shoulder_z
            normalized_lm.extend([normalized_x, normalized_y, normalized_z])

        # Normalize left arm landmarks
        normalized_lm.extend([left_shoulder_x, left_shoulder_y, left_shoulder_z])
        for lm_id in ['left_elbow', 'left_wrist', 'left_pinky', 'left_index', 'left_thumb']:
            landmark = landmarks[self.arm_landmarks[lm_id]]
            normalized_x = landmark.x - left_shoulder_x
            normalized_y = landmark.y - left_shoulder_y
            normalized_z = landmark.z - left_shoulder_z
            normalized_lm.extend([normalized_x, normalized_y, normalized_z])

        return normalized_lm

    def make_landmark_timestep(self, results):
        """Extract and normalize landmarks for a timestep."""
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        return self.normalize_landmarks(landmarks)

    def draw_class_on_image(self, img):
        """Display the predicted class label on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, self.label, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "L-Flexion: {}".format(str(self.repetition_left_flexion)), (510, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "R-Flexion: {}".format(str(self.repetition_right_flexion)), (510, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def detect_movement(self, lm_list):
        """Detect movement and update label."""
        lm_list = np.expand_dims(np.array(lm_list), axis=0)
        results = self.model.predict(lm_list)
        self.label = self.exercise_labels[np.argmax(results)]

    def process_frame(self, img, results):
        """Process each frame, extract landmarks, and predict movement."""
        img = self.draw_class_on_image(img)
        c_lm = self.make_landmark_timestep(results)
        if c_lm:
            self.lm_list.append(c_lm)
            if len(self.lm_list) == self.n_time_steps:
                threading.Thread(target=self.detect_movement, args=(self.lm_list,)).start()
                self.lm_list = self.lm_list[self.step_size:]
        return img

    def angle(self, img, LS, LE, LW, RS, RE, RW):

        if self.label == "FLEXION LEFT":
            cv2.line(img, (LS[0], 400), LS[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, LS[:2], LE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, LS[:2], 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, LE[:2], 5, (255, 0, 0), cv2.FILLED)
            reference = LS[0], 1000, LS[2]
            vector1 = np.array(reference) - np.array(LS)
            vector2 = np.array(LE) - np.array(LS)
        elif self.label == "FLEXION RIGHT":
            cv2.line(img, (RS[0], 400), RS[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, RS[:2], RE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, RS[:2], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, RE[:2], 5, (0, 0, 255), cv2.FILLED)
            reference = RS[0], 1000, RS[2]
            vector1 = np.array(reference) - np.array(RS)
            vector2 = np.array(RE) - np.array(RS)
        elif self.label == "CIRCLE LEFT":
            cv2.line(img, LW[:2], LW[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, LS[:2], LE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, LS[:2], 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, LE[:2], 5, (255, 0, 0), cv2.FILLED)
            reference = LS[0], 1000, LS[2]
            vector1 = np.array(reference) - np.array(LS)
            vector2 = np.array(LE) - np.array(LS)
        elif self.label == "CIRCLE RIGHT":
            cv2.line(img, RW[:2], RE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, RS[:2], RE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, RS[:2], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, RE[:2], 5, (0, 0, 255), cv2.FILLED)
            reference = RS[0], 1000, RS[2]
            vector1 = np.array(reference) - np.array(RS)
            vector2 = np.array(RE) - np.array(RS)
        else:
            reference = RS[0], 1000, RS[2]
            vector1 = np.array(reference) - np.array(RS)
            vector2 = np.array(RE) - np.array(RS)

        # Dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)
        cv2.putText(img, str(int(angle_degrees)), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return angle_degrees

    def get_state(self, angle):
        state = None

        # Dynamically fetch angle thresholds
        if self.label == "ARMPIT LEFT":
            threshold = self.angle_left_armpit
        elif self.label == "ARMPIT RIGHT":
            threshold = self.angle_right_armpit
        elif self.label == "CIRCLE LEFT":
            threshold = self.angle_left_circle
        elif self.label == "CIRCLE RIGHT":
            threshold = self.angle_right_circle
        elif self.label == "CB LEFT":
            threshold = self.angle_left_towel
        elif self.label == "CB RIGHT":
            threshold = self.angle_right_towel
        elif self.label == "PENDULUM LEFT":
            threshold = self.angle_left_towel
        elif self.label == "PENDULUM RIGHT":
            threshold = self.angle_right_towel
        elif self.label == "FLEXION LEFT":
            threshold = self.angle_left_flexion
        elif self.label == "FLEXION RIGHT":
            threshold = self.angle_right_flexion
        elif self.label == "TOWEL LEFT":
            threshold = self.angle_left_towel
        elif self.label == "TOWEL RIGHT":
            threshold = self.angle_right_towel
        else:
            threshold = 90  # Default threshold for undefined exercises

        # Define state ranges based on the dynamic threshold
        if 0 <= angle <= threshold - 60:
            state = 1
        elif threshold - 50 <= angle <= threshold - 10:
            state = 2
        elif threshold <= angle <= threshold + 10:
            state = 3
        return f"s{state}" if state else None

    def update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.list) and (self.list.count('s2')) == 0) or (
                    ('s3' in self.list) and (self.list.count('s2') == 1)):
                self.list.append(state)
        elif state == 's3':
            if (state not in self.list) and ('s2' in self.list):
                self.list.append(state)
        return self.list

    def counter(self, state):
        if state == 's1':
            if len(self.list) == 3:
                if self.label == "FLEXION LEFT":
                    self.repetition_left_flexion += 1
                elif self.label == "FLEXION RIGHT":
                    self.repetition_right_flexion += 1
                elif self.label == "CIRCLE LEFT":
                    self.repetition_left_cross += 1
                elif self.label == "CIRCLE RIGHT":
                    self.repetition_right_cross += 1
                else:
                    pass

            elif 's2' in self.list and len(self.list) == 1:
                print("Improper Form")

            self.list = []

        return self.repetition_left_flexion, self.repetition_right_flexion


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = int(w * r), height
    else:
        r = width / float(w)
        dim = width, int(h * r)

    # resizing of image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

MODEL = "NEW_CODE_V5.h5"

logo = "nyp_logo.png"
degree_of_movement = 0
### --------------------------------- STATE SESSIONS FOR REPETITION THRESHOLDS ------------------------------------- ###
# Armpit Stretch
if "repetition_threshold_left_armpit" not in st.session_state:
    st.session_state.repetition_threshold_left_armpit = 0
if "repetition_threshold_right_armpit" not in st.session_state:
    st.session_state.repetition_threshold_right_armpit = 0

# Arm Circles
if "repetition_threshold_left_circle" not in st.session_state:
    st.session_state.repetition_threshold_left_circle = 0
if "repetition_threshold_right_circle" not in st.session_state:
    st.session_state.repetition_threshold_right_circle = 0

# Cross Body Stretch
if "repetition_threshold_left_cross" not in st.session_state:
    st.session_state.repetition_threshold_left_cross = 0
if "repetition_threshold_right_cross" not in st.session_state:
    st.session_state.repetition_threshold_right_cross = 0

# Pendulum Swing
if "repetition_threshold_left_pendulum" not in st.session_state:
    st.session_state.repetition_threshold_left_pendulum = 0
if "repetition_threshold_right_pendulum" not in st.session_state:
    st.session_state.repetition_threshold_right_pendulum = 0

# Shoulder Flexion
if "repetition_threshold_left_flexion" not in st.session_state:
    st.session_state.repetition_threshold_left_flexion = 10
if "repetition_threshold_right_flexion" not in st.session_state:
    st.session_state.repetition_threshold_right_flexion = 10

# Towel Stretch
if "repetition_threshold_left_towel" not in st.session_state:
    st.session_state.repetition_threshold_left_towel = 0
if "repetition_threshold_right_towel" not in st.session_state:
    st.session_state.repetition_threshold_right_towel = 0

### -------------------------------------- STATE SESSIONS FOR SET THRESHOLDS --------------------------------------- ###
# Armpit Stretch
if "set_threshold_left_armpit" not in st.session_state:
    st.session_state.set_threshold_left_armpit = 0
if "set_threshold_right_armpit" not in st.session_state:
    st.session_state.set_threshold_right_armpit = 0

# Arm Circles
if "set_threshold_left_circle" not in st.session_state:
    st.session_state.set_threshold_left_circle = 0
if "set_threshold_right_circle" not in st.session_state:
    st.session_state.set_threshold_right_circle = 0

# Cross Body Stretch
if "set_threshold_left_cross" not in st.session_state:
    st.session_state.set_threshold_left_cross = 0
if "set_threshold_right_cross" not in st.session_state:
    st.session_state.set_threshold_right_cross = 0

# Pendulum Swing
if "set_threshold_left_pendulum" not in st.session_state:
    st.session_state.set_threshold_left_pendulum = 0
if "set_threshold_right_pendulum" not in st.session_state:
    st.session_state.set_threshold_right_pendulum = 0

# Shoulder Flexion
if "set_threshold_left_flexion" not in st.session_state:
    st.session_state.set_threshold_left_flexion = 10
if "set_threshold_right_flexion" not in st.session_state:
    st.session_state.set_threshold_right_flexion = 10

# Towel Stretch
if "set_threshold_left_towel" not in st.session_state:
    st.session_state.set_threshold_left_towel = 0
if "set_threshold_right_towel" not in st.session_state:
    st.session_state.set_threshold_right_towel = 0

### ------------------------------------ STATE SESSIONS FOR ANGLE THRESHOLDS --------------------------------------- ###
# Armpit Stretch
if "angle_left_armpit" not in st.session_state:
    st.session_state.angle_left_armpit = 90
if "angle_right_armpit" not in st.session_state:
    st.session_state.angle_right_armpit = 90

# Arm Circles
if "angle_left_circle" not in st.session_state:
    st.session_state.angle_left_circle = 90
if "angle_right_circle" not in st.session_state:
    st.session_state.angle_right_circle = 90

# Cross Body Stretch
if "angle_left_cross" not in st.session_state:
    st.session_state.angle_left_cross = 90
if "angle_right_cross" not in st.session_state:
    st.session_state.angle_right_cross = 90

# Pendulum Swing
if "angle_left_pendulum" not in st.session_state:
    st.session_state.angle_left_pendulum = 90
if "angle_right_pendulum" not in st.session_state:
    st.session_state.angle_right_pendulum = 90

# Shoulder Flexion
if "angle_left_flexion" not in st.session_state:
    st.session_state.angle_left_flexion = 90
if "angle_right_flexion" not in st.session_state:
    st.session_state.angle_right_flexion = 90

# Towel Stretch
if "angle_left_towel" not in st.session_state:
    st.session_state.angle_left_towel = 90
if "angle_right_towel" not in st.session_state:
    st.session_state.angle_right_towel = 90

### ------------------------------------ START OF USER INTERFACE CUSTOMISATIONS ------------------------------------ ###
st.logo(logo, icon_image=logo, size="large")
st.set_page_config(layout="wide")
st.title("Frozen Shoulder Rehabilitation Model")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px.,mn bv
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>


    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Frozen Shoulder Sidebar")
st.sidebar.subheader("Parameters")

st.cache_resource()

app_mode = st.sidebar.selectbox("Choose the App Mode", ["Target", "Video", "Angle"])

if app_mode == "Video":
    use_webcam = st.sidebar.toggle("Use Webcam")

    if use_webcam:


        vid = cv2.VideoCapture(0)
        detector = FrozenShoulder(MODEL)

        # Assign Repetition Thresholds from Session State to Detector (Alphabetical Order)
        detector.repetition_threshold_left_armpit = st.session_state.repetition_threshold_left_armpit
        detector.repetition_threshold_right_armpit = st.session_state.repetition_threshold_right_armpit
        detector.repetition_threshold_left_circle = st.session_state.repetition_threshold_left_circle
        detector.repetition_threshold_right_circle = st.session_state.repetition_threshold_right_circle
        detector.repetition_threshold_left_cross = st.session_state.repetition_threshold_left_cross
        detector.repetition_threshold_right_cross = st.session_state.repetition_threshold_right_cross
        detector.repetition_threshold_left_flexion = st.session_state.repetition_threshold_left_flexion
        detector.repetition_threshold_right_flexion = st.session_state.repetition_threshold_right_flexion
        detector.repetition_threshold_left_pendulum = st.session_state.repetition_threshold_left_pendulum
        detector.repetition_threshold_right_pendulum = st.session_state.repetition_threshold_right_pendulum
        detector.repetition_threshold_left_towel = st.session_state.repetition_threshold_left_towel
        detector.repetition_threshold_right_towel = st.session_state.repetition_threshold_right_towel

        # Assign Set Thresholds from Session State to Detector (Alphabetical Order)
        detector.set_threshold_left_armpit = st.session_state.set_threshold_left_armpit
        detector.set_threshold_right_armpit = st.session_state.set_threshold_right_armpit
        detector.set_threshold_left_circle = st.session_state.set_threshold_left_circle
        detector.set_threshold_right_circle = st.session_state.set_threshold_right_circle
        detector.set_threshold_left_cross = st.session_state.set_threshold_left_cross
        detector.set_threshold_right_cross = st.session_state.set_threshold_right_cross
        detector.set_threshold_left_flexion = st.session_state.set_threshold_left_flexion
        detector.set_threshold_right_flexion = st.session_state.set_threshold_right_flexion
        detector.set_threshold_left_pendulum = st.session_state.set_threshold_left_pendulum
        detector.set_threshold_right_pendulum = st.session_state.set_threshold_right_pendulum
        detector.set_threshold_left_towel = st.session_state.set_threshold_left_towel
        detector.set_threshold_right_towel = st.session_state.set_threshold_right_towel

        # Assign Angles Thresholds from Session State to Detector (Alphabetical Order)
        detector.angle_left_armpit = st.session_state.angle_left_armpit
        detector.angle_right_armpit = st.session_state.angle_right_armpit
        detector.angle_left_circle = st.session_state.angle_left_circle
        detector.angle_right_circle = st.session_state.angle_right_circle
        detector.angle_left_cross = st.session_state.angle_left_cross
        detector.angle_right_cross = st.session_state.angle_right_cross
        detector.angle_left_flexion = st.session_state.angle_left_flexion
        detector.angle_right_flexion = st.session_state.angle_right_flexion
        detector.angle_left_pendulum = st.session_state.angle_left_pendulum
        detector.angle_right_pendulum = st.session_state.angle_right_pendulum
        detector.angle_left_towel = st.session_state.angle_left_towel
        detector.angle_right_towel = st.session_state.angle_right_towel

        i = 0  # iterations
        warmup_frames = 60
        frame_count = 0

        # Create the layout
        c1, c2 = st.columns([0.7, 0.3], border=True)

        with c2:
            st.write(
                    """
                    <div style='text-align: center;'>
                        <h4 style='text-decoration: underline; font-weight: bold;'>TARGET</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            # Target
            if st.session_state.repetition_threshold_left_armpit > 0 or st.session_state.set_threshold_left_armpit > 0:
                text1 = st.markdown("")
            else:
                text1 = st.markdown("")

            if st.session_state.repetition_threshold_right_armpit > 0 or st.session_state.set_threshold_right_armpit > 0:
                text2 = st.markdown("")
            else:
                text2 = st.markdown("")

            # Arm Circles
            if st.session_state.repetition_threshold_left_circle > 0 or st.session_state.set_threshold_left_circle > 0:
                text3 = st.markdown("")
            else:
                text3 = st.markdown("")

            if st.session_state.repetition_threshold_right_circle > 0 or st.session_state.set_threshold_right_circle > 0:
                text4 = st.markdown("")
            else:
                text4 = st.markdown("")

            # Cross Body Stretch
            if st.session_state.repetition_threshold_left_cross > 0 or st.session_state.set_threshold_left_cross > 0:
                text5 = st.markdown("")
            else:
                text5 = st.markdown("")

            if st.session_state.repetition_threshold_right_cross > 0 or st.session_state.set_threshold_right_cross > 0:
                text6 = st.markdown("")
            else:
                text6 = st.markdown("")

            # Pendulum Swing
            if st.session_state.repetition_threshold_left_pendulum > 0 or st.session_state.set_threshold_left_pendulum > 0:
                text7 = st.markdown("")
            else:
                text7 = st.markdown("")

            if st.session_state.repetition_threshold_right_pendulum > 0 or st.session_state.set_threshold_right_pendulum > 0:
                text8 = st.markdown("")
            else:
                text8 = st.markdown("")

            # Shoulder Flexion
            if st.session_state.repetition_threshold_left_flexion > 0 or st.session_state.set_threshold_left_flexion > 0:
                text9 = st.markdown("")
            else:
                text9 = st.markdown("")

            if st.session_state.repetition_threshold_right_flexion > 0 or st.session_state.set_threshold_right_flexion > 0:
                text10 = st.markdown("")
            else:
                text10 = st.markdown("")

            # Towel Stretch
            if st.session_state.repetition_threshold_left_towel > 0 or st.session_state.set_threshold_left_towel > 0:
                text11 = st.markdown("")
            else:
                text11 = st.markdown("")

            if st.session_state.repetition_threshold_right_towel > 0 or st.session_state.set_threshold_right_towel > 0:
                text12 = st.markdown("")
            else:
                text12 = st.markdown("")


        # Video display goes in c1
        with c1:
            exercise_label = st.markdown("Type of Exercise")
            stframe = st.empty()  # Video frame placeholder

        # Process video frames
        while vid.isOpened():
            i += 1
            success, img = vid.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.pose.process(imgRGB)
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)
            if len(lmList) != 0:
                degree_of_movement = detector.angle(img, lmList[11][1:], lmList[13][1:], lmList[15][1:], lmList[12][1:], lmList[14][1:], lmList[16][1:])
                current_state = detector.get_state(degree_of_movement)
                detector.update_state_sequence(current_state)
                detector.counter(current_state)

                # Update the counters in the UI
                with c2:
                    # Target shit
                    if detector.label == "ARMPIT LEFT":
                        color_left_armpit = "red"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "ARMPIT RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "red"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "CIRCLE LEFT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "red"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "CIRCLE RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "red"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "CB LEFT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "red"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "CB RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "red"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "FLEXION LEFT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "red"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "FLEXION RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "red"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "PENDULUM LEFT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "red"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "PENDULUM RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "red"
                        color_left_towel = "white"
                        color_right_towel = "white"
                    elif detector.label == "TOWEL LEFT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "red"
                        color_right_towel = "white"
                    elif detector.label == "TOWEL RIGHT":
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "red"
                    else:
                        color_left_armpit = "white"
                        color_right_armpit = "white"
                        color_left_circle = "white"
                        color_right_circle = "white"
                        color_left_cross = "white"
                        color_right_cross = "white"
                        color_left_flexion = "white"
                        color_right_flexion = "white"
                        color_left_pendulum = "white"
                        color_right_pendulum = "white"
                        color_left_towel = "white"
                        color_right_towel = "white"


                    # Armpit Stretch
                    if st.session_state.repetition_threshold_left_armpit > 0 or st.session_state.set_threshold_left_armpit > 0:
                        text1.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_armpit};'>> Left Armpit Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_armpit}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_armpit} / {st.session_state.repetition_threshold_left_armpit}</span>
                                <span style='color: {color_left_armpit}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_armpit} / {st.session_state.set_threshold_left_armpit}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text1.write("")

                    if st.session_state.repetition_threshold_right_armpit > 0 or st.session_state.set_threshold_right_armpit > 0:
                        text2.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_armpit};'>> Right Armpit Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_armpit}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_armpit} / {st.session_state.repetition_threshold_right_armpit}</span>
                                <span style='color: {color_right_armpit}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_armpit} / {st.session_state.set_threshold_right_armpit}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text2.write("")

                    # Arm Circles
                    if st.session_state.repetition_threshold_left_circle > 0 or st.session_state.set_threshold_left_circle > 0:
                        text3.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_circle};'>> Left Arm Circles</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_circle}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_circle} / {st.session_state.repetition_threshold_left_circle}</span>
                                <span style='color: {color_left_circle}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_circle} / {st.session_state.set_threshold_left_circle}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text3.write("")

                    if st.session_state.repetition_threshold_right_circle > 0 or st.session_state.set_threshold_right_circle > 0:
                        text4.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_circle};'>> Right Arm Circles</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_circle}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_circle} / {st.session_state.repetition_threshold_right_circle}</span>
                                <span style='color: {color_right_circle}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_circle} / {st.session_state.set_threshold_right_circle}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text4.write("")

                    # Cross Body Stretch
                    if st.session_state.repetition_threshold_left_cross > 0 or st.session_state.set_threshold_left_cross > 0:
                        text5.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_cross};'>> Left Cross Body Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_cross}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_cross} / {st.session_state.repetition_threshold_left_cross}</span>
                                <span style='color: {color_left_cross}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_cross} / {st.session_state.set_threshold_left_cross}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text5.write("")

                    if st.session_state.repetition_threshold_right_cross > 0 or st.session_state.set_threshold_right_cross > 0:
                        text6.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_cross};'>> Right Cross Body Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_cross}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_cross} / {st.session_state.repetition_threshold_right_cross}</span>
                                <span style='color: {color_right_cross}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_cross} / {st.session_state.set_threshold_right_cross}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text6.write("")

                    # Pendulum Swing
                    if st.session_state.repetition_threshold_left_pendulum > 0 or st.session_state.set_threshold_left_pendulum > 0:
                        text7.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_pendulum};'>> Left Pendulum Swing</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_pendulum}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_pendulum} / {st.session_state.repetition_threshold_left_pendulum}</span>
                                <span style='color: {color_left_pendulum}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_pendulum} / {st.session_state.set_threshold_left_pendulum}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text7.write("")

                    if st.session_state.repetition_threshold_right_pendulum > 0 or st.session_state.set_threshold_right_pendulum > 0:
                        text8.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_pendulum};'>> Right Pendulum Swing</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_pendulum}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_pendulum} / {st.session_state.repetition_threshold_right_pendulum}</span>
                                <span style='color: {color_right_pendulum}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_pendulum} / {st.session_state.set_threshold_right_pendulum}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text8.write("")

                    # Shoulder Flexion
                    if st.session_state.repetition_threshold_left_flexion > 0 or st.session_state.set_threshold_left_flexion > 0:
                        text9.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_flexion};'>> Left Shoulder Flexion</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_flexion}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_flexion} / {st.session_state.repetition_threshold_left_flexion}</span>
                                <span style='color: {color_left_flexion}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_flexion} / {st.session_state.set_threshold_left_flexion}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text9.write("")

                    if st.session_state.repetition_threshold_right_flexion > 0 or st.session_state.set_threshold_right_flexion > 0:
                        text10.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_flexion};'>> Right Shoulder Flexion</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_flexion}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_flexion} / {st.session_state.repetition_threshold_right_flexion}</span>
                                <span style='color: {color_right_flexion}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_flexion} / {st.session_state.set_threshold_right_flexion}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text10.write("")

                    # Towel Stretch
                    if st.session_state.repetition_threshold_left_towel > 0 or st.session_state.set_threshold_left_towel > 0:
                        text11.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_left_towel};'>> Left Towel Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_left_towel}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_left_towel} / {st.session_state.repetition_threshold_left_towel}</span>
                                <span style='color: {color_left_towel}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_left_towel} / {st.session_state.set_threshold_left_towel}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text11.write("")

                    if st.session_state.repetition_threshold_right_towel > 0 or st.session_state.set_threshold_right_towel > 0:
                        text12.write(
                            f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: {color_right_towel};'>> Right Towel Stretch</span>
                            <div style='display: flex; gap: 20px;'>
                                <span style='color: {color_right_towel}; font-size: 16px; font-weight: bold;'>Reps: {detector.repetition_right_towel} / {st.session_state.repetition_threshold_right_towel}</span>
                                <span style='color: {color_right_towel}; font-size: 16px; font-weight: bold;'>Sets: {detector.set_right_towel} / {st.session_state.set_threshold_right_towel}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        text12.write("")

            # Display the processed frame in c1
            imgRGB = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
            imgRGB = image_resize(image=img, width=640)
            with c1:
                # Dynamically fetch the threshold value for the current exercise
                if detector.label == "ARMPIT LEFT":
                    threshold = detector.angle_left_armpit
                elif detector.label == "ARMPIT RIGHT":
                    threshold = detector.angle_right_armpit
                elif detector.label == "CIRCLE LEFT":
                    threshold = detector.angle_left_circle
                elif detector.label == "CIRCLE RIGHT":
                    threshold = detector.angle_right_circle
                elif detector.label == "CB LEFT":
                    threshold = detector.angle_left_towel
                elif detector.label == "CB RIGHT":
                    threshold = detector.angle_right_towel
                elif detector.label == "PENDULUM LEFT":
                    threshold = detector.angle_left_towel
                elif detector.label == "PENDULUM RIGHT":
                    threshold = detector.angle_right_towel
                elif detector.label == "FLEXION LEFT":
                    threshold = detector.angle_left_flexion
                elif detector.label == "FLEXION RIGHT":
                    threshold = detector.angle_right_flexion
                elif detector.label == "TOWEL LEFT":
                    threshold = detector.angle_left_towel
                elif detector.label == "TOWEL RIGHT":
                    threshold = detector.angle_right_towel
                else:
                    threshold = 90  # Default threshold for undefined exercises

                # Display the current exercise, ROM, and threshold
                exercise_label.write(
                    f"""
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='color: red;'>Current Exercise: {detector.label}</h3>
                            <h3 style='color: blue;'>ROM: {int(degree_of_movement)}° / Threshold: {threshold}°</h3>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )
                stframe.image(imgRGB, channels="BGR", use_container_width=True)

            frame_count += 1
            if frame_count > warmup_frames:
                img = detector.process_frame(img, results)
    else:
        pass

elif app_mode == "Target":
    st.empty()
    st.header("Set Exercise Targets")
    T1, T2, T3 = st.columns(3, border=True)
    with T1:
        # Armpit Stretch Targets
        st.subheader("Armpit Stretch")
        st.session_state.repetition_threshold_left_armpit = st.number_input(
            "Repetitions Per Set for Left Armpit Stretch",
            step=1,
            value=st.session_state.repetition_threshold_left_armpit,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_armpit = st.number_input(
            "Target Sets for Left Armpit Stretch",
            step=1,
            value=st.session_state.set_threshold_left_armpit,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_armpit = st.number_input(
            "Repetitions Per Set for Right Armpit Stretch",
            step=1,
            value=st.session_state.repetition_threshold_right_armpit,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_armpit = st.number_input(
            "Target Sets for Right Armpit Stretch",
            step=1,
            value=st.session_state.set_threshold_right_armpit,
            placeholder="Enter Amount..."
        )
        pass
    with T2:
        # Arm Circles Targets
        st.subheader("Arm Circles")
        st.session_state.repetition_threshold_left_circle = st.number_input(
            "Repetitions Per Set for Left Arm Circles",
            step=1,
            value=st.session_state.repetition_threshold_left_circle,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_circle = st.number_input(
            "Target Sets for Left Arm Circles",
            step=1,
            value=st.session_state.set_threshold_left_circle,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_circle = st.number_input(
            "Repetitions Per Set for Right Arm Circles",
            step=1,
            value=st.session_state.repetition_threshold_right_circle,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_circle = st.number_input(
            "Target Sets for Right Arm Circles",
            step=1,
            value=st.session_state.set_threshold_right_circle,
            placeholder="Enter Amount..."
        )
        pass
    with T3:
        # Cross Body Stretch Targets
        st.subheader("Cross Body Stretch")
        st.session_state.repetition_threshold_left_cross = st.number_input(
            "Repetitions Per Set for Left Cross Body Stretch",
            step=1,
            value=st.session_state.repetition_threshold_left_cross,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_cross = st.number_input(
            "Target Sets for Left Cross Body Stretch",
            step=1,
            value=st.session_state.set_threshold_left_cross,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_cross = st.number_input(
            "Repetitions Per Set for Right Cross Body Stretch",
            step=1,
            value=st.session_state.repetition_threshold_right_cross,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_cross = st.number_input(
            "Target Sets for Right Cross Body Stretch",
            step=1,
            value=st.session_state.set_threshold_right_cross,
            placeholder="Enter Amount..."
        )
        pass
    T4, T5, T6 = st.columns(3, border=True)
    with T4:
        # Pendulum Swing Targets
        st.subheader("Pendulum Swing")
        st.session_state.repetition_threshold_left_pendulum = st.number_input(
            "Repetitions Per Set for Left Pendulum Swing",
            step=1,
            value=st.session_state.repetition_threshold_left_pendulum,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_pendulum = st.number_input(
            "Target Sets for Left Pendulum Swing",
            step=1,
            value=st.session_state.set_threshold_left_pendulum,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_pendulum = st.number_input(
            "Repetitions Per Set for Right Pendulum Swing",
            step=1,
            value=st.session_state.repetition_threshold_right_pendulum,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_pendulum = st.number_input(
            "Target Sets for Right Pendulum Swing",
            step=1,
            value=st.session_state.set_threshold_right_pendulum,
            placeholder="Enter Amount..."
        )
        pass
    with T5:
        # Shoulder Flexion Targets
        st.subheader("Shoulder Flexion")
        st.session_state.repetition_threshold_left_flexion = st.number_input(
            "Repetitions Per Set for Left Shoulder Flexion",
            step=1,
            value=st.session_state.repetition_threshold_left_flexion,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_flexion = st.number_input(
            "Target Sets for Left Shoulder Flexion",
            step=1,
            value=st.session_state.set_threshold_left_flexion,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_flexion = st.number_input(
            "Repetitions Per Set for Right Shoulder Flexion",
            step=1,
            value=st.session_state.repetition_threshold_right_flexion,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_flexion = st.number_input(
            "Target Sets for Right Shoulder Flexion",
            step=1,
            value=st.session_state.set_threshold_right_flexion,
            placeholder="Enter Amount..."
        )
        pass
    with T6:
        # Towel Stretch Targets
        st.subheader("Towel Stretch")
        st.session_state.repetition_threshold_left_towel = st.number_input(
            "Repetitions Per Set for Left Towel Stretch",
            step=1,
            value=st.session_state.repetition_threshold_left_towel,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_left_towel = st.number_input(
            "Target Sets for Left Towel Stretch",
            step=1,
            value=st.session_state.set_threshold_left_towel,
            placeholder="Enter Amount..."
        )
        st.divider()
        st.session_state.repetition_threshold_right_towel = st.number_input(
            "Repetitions Per Set for Right Towel Stretch",
            step=1,
            value=st.session_state.repetition_threshold_right_towel,
            placeholder="Enter Amount..."
        )
        st.session_state.set_threshold_right_towel = st.number_input(
            "Target Sets for Right Towel Stretch",
            step=1,
            value=st.session_state.set_threshold_right_towel,
            placeholder="Enter Amount..."
        )
        pass

elif app_mode == "Angle":
    st.empty()
    st.header("Set Exercise Angles")
    T1, T2, T3 = st.columns(3, border=True)
    with T1:
        # Armpit Stretch Angles
        st.subheader("Armpit Stretch")
        st.session_state.angle_left_armpit = st.number_input(
            "Angle Repetitions for Left Armpit Stretch",
            step=1,
            value=st.session_state.angle_left_armpit,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_armpit = st.number_input(
            "Angle Repetitions for Right Armpit Stretch",
            step=1,
            value=st.session_state.angle_right_armpit,
            placeholder="Enter Amount..."
        )
        pass
    with T2:
        # Arm Circles Angles
        st.subheader("Arm Circles")
        st.session_state.angle_left_circle = st.number_input(
            "Angle Repetitions for Left Arm Circles",
            step=1,
            value=st.session_state.angle_left_circle,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_circle = st.number_input(
            "Angle Repetitions for Right Arm Circles",
            step=1,
            value=st.session_state.angle_right_circle,
            placeholder="Enter Amount..."
        )
        pass
    with T3:
        # Cross Body Stretch Angles
        st.subheader("Cross Body Stretch")
        st.session_state.angle_left_cross = st.number_input(
            "Angle Repetitions for Left Cross Body Stretch",
            step=1,
            value=st.session_state.angle_left_cross,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_cross = st.number_input(
            "Angle Repetitions for Right Cross Body Stretch",
            step=1,
            value=st.session_state.angle_right_cross,
            placeholder="Enter Amount..."
        )
        pass
    T4, T5, T6 = st.columns(3, border=True)
    with T4:
        # Pendulum Swing Angles
        st.subheader("Pendulum Swing")
        st.session_state.angle_left_pendulum = st.number_input(
            "Angle Repetitions for Left Pendulum Swing",
            step=1,
            value=st.session_state.angle_left_pendulum,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_pendulum = st.number_input(
            "Angle Repetitions for Right Pendulum Swing",
            step=1,
            value=st.session_state.angle_right_pendulum,
            placeholder="Enter Amount..."
        )
        pass
    with T5:
        # Shoulder Flexion Angles
        st.subheader("Shoulder Flexion")
        st.session_state.angle_left_flexion = st.number_input(
            "Angle Repetitions for Left Shoulder Flexion",
            step=1,
            value=st.session_state.angle_left_flexion,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_flexion = st.number_input(
            "Angle Repetitions for Right Shoulder Flexion",
            step=1,
            value=st.session_state.angle_right_flexion,
            placeholder="Enter Amount..."
        )
        pass
    with T6:
        # Towel Stretch Angles
        st.subheader("Towel Stretch")
        st.session_state.angle_left_towel = st.number_input(
            "Angle Repetitions for Left Towel Stretch",
            step=1,
            value=st.session_state.angle_left_towel,
            placeholder="Enter Amount..."
        )
        st.session_state.angle_right_towel = st.number_input(
            "Angle Repetitions for Right Towel Stretch",
            step=1,
            value=st.session_state.angle_right_towel,
            placeholder="Enter Amount..."
        )
        pass
else:
    pass