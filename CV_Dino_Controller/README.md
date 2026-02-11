# 🦖 CV Dino Controller

Control the Google Chrome "Dino" game using hand gestures!

## 📋 Prerequisites

1.  **Python 3.8+** installed.
2.  **Google Chrome** browser installed.
3.  A working **Webcam**.

## 🚀 Setup & Installation

1.  Open your terminal/command prompt.
2.  Navigate to this project folder:
    ```bash
    cd CV_Dino_Controller
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 How to Run

1.  Run the main script:
    ```bash
    python dino_control.py
    ```
2.  **Wait a moment**:
    *   A Chrome window will open automatically and load the Dino game.
    *   Your webcam feed will appear in a separate window.

## 🖐️ Controls

*   **JUMP**: Raise **EXACTLY 1 Finger** (Index Finger).
*   **Idle**: Keep your hand open or closed (fist).
*   **Quit**: Click on the webcam window and press **'q'** to stop.

## 🛠️ Troubleshooting

*   **Browser not opening?** Ensure Chrome is installed. The script uses `webdriver-manager` to automatically download the correct driver.
*   **Game not jumping?** Make sure the browser window is **focused** (click on it once if needed).
*   **Hand not detected?** Ensure your hand is visible in the webcam frame and there is decent lighting.
