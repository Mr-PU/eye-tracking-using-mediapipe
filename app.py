import streamlit as st
import numpy as np
import cv2
import time
import pygame
from FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils
import os

# Initialize pygame mixer for audio
pygame.mixer.init()

# Suppress TensorFlow oneDNN log (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Uncomment to disable oneDNN if needed
# Suppress TensorFlow logging (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

class EyeStateDetector:
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]
    
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self, threshold):
        self.generator = FaceMeshGenerator(
            mode=False,
            num_faces=1,
            min_detection_con=0.5,
            min_track_con=0.5
        )
        self.EAR_THRESHOLD = threshold
        self.eyes_closed_start_time = None
        self.alert_played = False
        self.ear_history = []
        self.history_size = 10
        self.hysteresis_threshold = 0.02
        self.current_state = "Open"
        self.calibration_samples = None
        # self.alert_sound = pygame.mixer.Sound(
        #     r"media/emergency-siren-alert-single-epic-stock-media-1-00-01.mp3"
        # )
        try:
            self.alert_sound = pygame.mixer.Sound(
                r"media/emergency-siren-alert-single-epic-stock-media-1-00-01.mp3"
            )
        except Exception as e:
            st.error(f"Failed to load audio file: {str(e)}")

    def enable_calibration(self):
        if self.calibration_samples is None:
            self.calibration_samples = {'open': [], 'closed': []}

    def disable_calibration(self):
        self.calibration_samples = None

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - 
                          np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - 
                          np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - 
                          np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def smooth_ear(self, ear):
        self.ear_history.append(ear)
        if len(self.ear_history) > self.history_size:
            self.ear_history.pop(0)
        return np.mean(self.ear_history)

    def determine_eye_state(self, ear):
        if self.current_state == "Open" and ear < self.EAR_THRESHOLD - self.hysteresis_threshold:
            self.current_state = "Closed"
        elif self.current_state == "Closed" and ear > self.EAR_THRESHOLD + self.hysteresis_threshold:
            self.current_state = "Open"
        return self.current_state

    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Frame is empty or invalid")
            frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
            if not face_landmarks:
                return frame, None, None
            
            right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
            left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
            ear = (right_ear + left_ear) / 2.0
            
            smoothed_ear = self.smooth_ear(ear)
            eye_state = self.determine_eye_state(smoothed_ear)
            
            DrawingUtils.draw_text_with_bg(
                frame, f"Eyes: {eye_state}", (0, 60),
                font_scale=2, thickness=3,
                bg_color=self.COLORS['GREEN']['bgr'], text_color=(0, 0, 0)
            )
            
            return frame, smoothed_ear, eye_state
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return frame, None, None

    def calibrate(self, frame, state):
        if self.calibration_samples is None:
            self.enable_calibration()
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Frame is empty or invalid")
            frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
            if not face_landmarks:
                return None
            right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
            left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
            ear = (right_ear + left_ear) / 2.0
            self.calibration_samples[state].append(ear)
            return ear
        except Exception as e:
            st.error(f"Calibration error: {str(e)}")
            return None

    def finalize_calibration(self):
        if self.calibration_samples and self.calibration_samples['open'] and self.calibration_samples['closed']:
            open_avg = np.mean(self.calibration_samples['open'])
            closed_avg = np.mean(self.calibration_samples['closed'])
            self.EAR_THRESHOLD = (open_avg + closed_avg) / 2.0
            self.hysteresis_threshold = (open_avg - closed_avg) / 4.0
            return self.EAR_THRESHOLD, self.hysteresis_threshold
        return None, None

# Function to process and display a single frame
def process_and_display_frame(detector, cap, frame_placeholder, alert_duration, calibrate_mode):
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        st.error("Error: Failed to capture frame from webcam.")
        return False, None
    
    if not calibrate_mode:
        processed_frame, ear, eye_state = detector.process_frame(frame)
        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

        if eye_state == "Closed":
            if detector.eyes_closed_start_time is None:
                detector.eyes_closed_start_time = time.time()
            else:
                closed_duration = time.time() - detector.eyes_closed_start_time
                if closed_duration >= alert_duration and not detector.alert_played:
                    st.warning(f"Eyes closed for more than {alert_duration} seconds! Playing alert...")
                    detector.alert_sound.play()
                    detector.alert_played = True
        else:
            detector.eyes_closed_start_time = None
            detector.alert_played = False
        return True, processed_frame
    else:
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        return True, frame

# Streamlit App
def main():
    st.title("Live Eye State Detection using Webcam")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.5, 0.21, 0.01)
    alert_duration = st.sidebar.slider("Alert Duration (seconds)", 1, 10, 2)
    calibrate_mode = st.sidebar.checkbox("Enable Calibration Mode")

    # Initialize session state
    if 'eye_detector' not in st.session_state:
        st.session_state.eye_detector = EyeStateDetector(threshold=ear_threshold)
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("Error: Could not open webcam.")
            st.stop()
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

    # Update EAR threshold dynamically
    st.session_state.eye_detector.EAR_THRESHOLD = ear_threshold

    # Update calibration mode
    if calibrate_mode:
        st.session_state.eye_detector.enable_calibration()
    else:
        st.session_state.eye_detector.disable_calibration()

    # Start/Stop button
    if st.button("Start/Stop Detection"):
        st.session_state.running = not st.session_state.running
        if st.session_state.running and not st.session_state.cap.isOpened():
            st.session_state.cap.release()
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Error: Could not reinitialize webcam.")
                st.session_state.running = False
        st.session_state.last_update_time = time.time()

    # Status display
    st.sidebar.write(f"Streaming: {st.session_state.running}")

    frame_placeholder = st.empty()

    # Calibration controls
    if calibrate_mode:
        st.write("Calibration Mode: Sample your eye states.")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Sample Open Eyes"):
                ret, frame = st.session_state.cap.read()
                if ret and frame is not None and frame.size > 0:
                    ear = st.session_state.eye_detector.calibrate(frame, 'open')
                    if ear:
                        st.write(f"Open EAR: {ear:.3f}")
                    else:
                        st.write("Failed to detect face landmarks.")
                else:
                    st.write("Failed to capture frame.")
        with col2:
            if st.button("Sample Closed Eyes"):
                ret, frame = st.session_state.cap.read()
                if ret and frame is not None and frame.size > 0:
                    ear = st.session_state.eye_detector.calibrate(frame, 'closed')
                    if ear:
                        st.write(f"Closed EAR: {ear:.3f}")
                    else:
                        st.write("Failed to detect face landmarks.")
                else:
                    st.write("Failed to capture frame.")
        with col3:
            if st.button("Finalize Calibration"):
                threshold, hysteresis = st.session_state.eye_detector.finalize_calibration()
                if threshold:
                    st.write(f"Calibrated EAR Threshold: {threshold:.3f}")
                    st.write(f"Updated Hysteresis: {hysteresis:.3f}")
                else:
                    st.write("Calibration incomplete: Sample both open and closed eyes.")

    # Continuous frame processing
    current_time = time.time()
    if st.session_state.running:
        if current_time - st.session_state.last_update_time >= 0.05:  # ~30 FPS
            success, frame = process_and_display_frame(
                st.session_state.eye_detector,
                st.session_state.cap,
                frame_placeholder,
                alert_duration,
                calibrate_mode
            )
            if success:
                st.session_state.last_frame = frame
                st.session_state.last_update_time = current_time
            else:
                st.session_state.running = False
    else:
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
        else:
            ret, frame = st.session_state.cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                st.session_state.last_frame = frame

    # Trigger rerun for continuous streaming when running
    if st.session_state.running:
        time.sleep(0.2)
        st.rerun()

if __name__ == "__main__":
    main()

def cleanup():
    if 'cap' in st.session_state:
        st.session_state.cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()