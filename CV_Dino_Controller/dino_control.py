import cv2
import mediapipe as mp
import time
import sys
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# --- 1. SETUP & CONFIGURATION ---
print("🦖 Initializing Dino Controller (MediaPipe Tasks API)...")

# Verify Model Exists
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    print("👉 Please download it: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    sys.exit(1)

# Initialize Selenium
try:
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    DINO_URL = "https://chromedino.com/"
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(DINO_URL)
    print("🌐 Browser Launched!")
    
    # Initial Focus
    time.sleep(2)
    body = driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.SPACE)
    print("🎮 Game Started!")
    
except Exception as e:
    print(f"❌ Error launching browser: {e}")
    # Continue even if browser fails for testing CV
    driver = None

# Global Variables for Async Callback
current_fingers = 0
jump_status = "Idle"
jump_color = (0, 255, 0)
last_jump_time = 0
latest_result = None

# --- MEDIAPIPE TASKS SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle results
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

landmarker = HandLandmarker.create_from_options(options)

# --- FINGER COUNTING LOGIC ---
def count_fingers_from_landmarks(landmarks):
    # landmarks is a list of NormalizedLandmark
    if not landmarks: return 0, []
    
    fingers = []
    
    # Thumb: x-axis comparison (assuming right hand, adjust logic if needed)
    # Tip (4) vs IP Joint (3). 
    # Logic: For simplicity in mirror mode, we check relative x difference
    # Ideally: Check if Tip is "further out" than Knuckle.
    # Let's use a simple distance check or x-check relative to wrist for robustness?
    # Simple X-check:
    if landmarks[4].x > landmarks[3].x: # Right hand facing camera -> Thumb is right
        fingers.append(1)
    else:
        fingers.append(0) # Logic might flip depending on hand, but Index is main trigger
    
    # 4 Fingers: Y-axis (Tip < PIP)
    # Y increases downwards
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky Tips
    pips = [6, 10, 14, 18] # PIP joints
    
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers.count(1), fingers

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("📷 Starting Webcam Feed... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocess
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Send to MediaPipe (Async)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)
        
        # Process Results (if available from callback)
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw landmarks manually (simple dots/lines)
                h, w, c = frame.shape
                
                # Draw Connections
                # (Simple loop or use mp_drawing if available, but mp_drawing needs 'solutions' which is broken)
                # We'll just draw points for the tips to keep it simple and crash-free
                
                # Get Coordinates
                lm_list = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                # Count Fingers
                total_fingers, finger_states = count_fingers_from_landmarks(hand_landmarks)
                
                # --- JUMP TRIGGER ---
                # Rule: Exactly 1 Finger (Index) is UP
                # finger_states: [Thumb, Index, Middle, Ring, Pinky]
                is_jump = (finger_states[1] == 1 and finger_states[2] == 0 and finger_states[3] == 0 and finger_states[4] == 0)
                
                current_time = time.time()
                if is_jump:
                    if (current_time - last_jump_time) > 0.3: # Cooldown
                        jump_status = "JUMP!"
                        jump_color = (0, 0, 255)
                        last_jump_time = current_time
                        
                        if driver:
                            try:
                                body.send_keys(Keys.SPACE)
                            except:
                                pass
                        
                        # Highlight Index Tip
                        idx_x, idx_y = lm_list[8]
                        cv2.circle(frame, (idx_x, idx_y), 15, (0, 0, 255), cv2.FILLED)
                else:
                    jump_status = "Idle"
                    jump_color = (0, 255, 0)

                # Overlay Text
                cv2.rectangle(frame, (20, 20), (200, 100), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, f"Fingers: {total_fingers}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, jump_status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, jump_color, 2)

        cv2.imshow("CV Dino Controller (Tasks API)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    if driver:
        driver.quit()
