import streamlit as st
import numpy as np
import cv2
import time
from playsound import playsound
from FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils

class EyeStateDetector:
    """
    A class to detect whether eyes are open or closed using facial landmarks.
    """
    
    # Define facial landmark indices for eyes
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]  # Points for EAR calculation
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]  # Points for EAR calculation
    
    # Define colors for visualization
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self, threshold):
        """
        Initialize the EyeStateDetector with detection parameters.
        
        Args:
            threshold (float): EAR threshold for eye state detection
        """
        # Initialize core parameters
        self.generator = FaceMeshGenerator()
        self.EAR_THRESHOLD = threshold

        # Timer for tracking closed eyes duration
        self.eyes_closed_start_time = None
        self.alert_played = False

        # Smoothing and hysteresis parameters
        self.ear_history = []  # Stores the last N EAR values for smoothing
        self.history_size = 10  # Number of frames to average for smoothing
        self.hysteresis_threshold = 0.02  # Hysteresis range to prevent rapid toggling
        self.current_state = "Open"  # Current eye state

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        """
        Calculate the eye aspect ratio (EAR) for given eye landmarks.
        """
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - 
                          np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - 
                          np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - 
                          np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def smooth_ear(self, ear):
        """
        Smooth the EAR value using a moving average.
        """
        self.ear_history.append(ear)
        if len(self.ear_history) > self.history_size:
            self.ear_history.pop(0)
        return np.mean(self.ear_history)

    def determine_eye_state(self, ear):
        """
        Determine the eye state using hysteresis to prevent rapid toggling.
        """
        if self.current_state == "Open" and ear < self.EAR_THRESHOLD - self.hysteresis_threshold:
            self.current_state = "Closed"
        elif self.current_state == "Closed" and ear > self.EAR_THRESHOLD + self.hysteresis_threshold:
            self.current_state = "Open"
        return self.current_state

    def process_frame(self, frame):
        """
        Process a single frame to detect and analyze eyes.
        """
        frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
        
        if not face_landmarks:
            return frame, None, None
            
        # Calculate EAR
        right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
        left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
        ear = (right_ear + left_ear) / 2.0
        
        # Smooth the EAR value
        smoothed_ear = self.smooth_ear(ear)
        
        # Determine eye state with hysteresis
        eye_state = self.determine_eye_state(smoothed_ear)
        
        # Draw eye state on the frame
        DrawingUtils.draw_text_with_bg(
            frame, f"Eyes: {eye_state}", (0, 60),
            font_scale=2, thickness=3,
            bg_color=self.COLORS['GREEN']['bgr'], text_color=(0, 0, 0)
        )
        
        return frame, smoothed_ear, eye_state

# Streamlit App
st.title("Live Eye State Detection using Webcam")

# Sidebar for configuration
st.sidebar.header("Configuration")
ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.3, 0.21, 0.01)
alert_duration = st.sidebar.slider("Alert Duration (seconds)", 1, 10, 2)

# Initialize EyeStateDetector
eye_state_detector = EyeStateDetector(threshold=ear_threshold)

# Placeholder for displaying the live stream
frame_placeholder = st.empty()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Main loop for live stream processing
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture frame.")
        break

    # Process the frame
    processed_frame, ear, eye_state = eye_state_detector.process_frame(frame)

    # Display the processed frame
    frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

    # Check if eyes are closed
    if eye_state == "Closed":
        if eye_state_detector.eyes_closed_start_time is None:
            # Start the timer
            eye_state_detector.eyes_closed_start_time = time.time()
        else:
            # Calculate the duration of closed eyes
            closed_duration = time.time() - eye_state_detector.eyes_closed_start_time
            if closed_duration >= alert_duration and not eye_state_detector.alert_played:
                # Play audio alert
                st.warning("Eyes closed for more than 2 seconds! Playing alert...")
                playsound(r"/home/prashant/Documents/work-related/projects/test/yolov8/Eye-Blink-Detection-using-MediaPipe-and-OpenCV-master/emergency-siren-alert-single-epic-stock-media-1-00-01.mp3")
                eye_state_detector.alert_played = True
    else:
        # Reset the timer and alert flag
        eye_state_detector.eyes_closed_start_time = None
        eye_state_detector.alert_played = False

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


