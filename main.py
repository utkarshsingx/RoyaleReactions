import cv2
import os
import time
import random
import threading
from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier
import numpy as np

# Try to import pygame for audio playback
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(8) # Allow up to 8 simultaneous sounds
    AUDIO_AVAILABLE = True
    print("Audio support enabled.")
except ImportError:
    AUDIO_AVAILABLE = False
    print("Pygame not available - audio disabled. Install with: pip install pygame")

def load_reference_images():
    """Load reference images for each pose class"""
    images_dir = "images"
    reference_images = {}
    
    # Map pose names to image filenames
    pose_images = {
        "Laughing": "laughing.png",
        "Yawning": "yawning.png", 
        "Crying": "crying.png",
        "Taunting": "taunting.png"
    }
    
    for pose_name, filename in pose_images.items():
        image_path = os.path.join(images_dir, filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                reference_images[pose_name] = img
                print(f"Loaded reference image for {pose_name}: {filename}")
            else:
                print(f"Could not load image: {image_path}")
        else:
            print(f"Reference image not found: {image_path}")
    
    return reference_images

def play_pose_sound(pose_name):
    """Play sound for the detected pose (overlapping with previous sounds)"""
    if not AUDIO_AVAILABLE:
        return
    
    try:
        # Map pose names to sound files
        sound_files = {
            "Laughing": "laughing.mp3",
            "Yawning": "yawning.mp3",
            "Crying": "crying.mp3",
            "Taunting": "taunting.mp3"
        }
        
        if pose_name in sound_files:
            sound_path = os.path.join("sounds", sound_files[pose_name])
            if os.path.exists(sound_path):
                def play_sound():
                    try:
                        sound = pygame.mixer.Sound(sound_path)
                        sound.play()
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                
                sound_thread = threading.Thread(target=play_sound)
                sound_thread.daemon = True
                sound_thread.start()
            else:
                print(f"Sound file not found: {sound_path}")
    except Exception as e:
        print(f"Error with audio: {e}")

def show_reference_image(pose_name, reference_images, window_name="Reference Image"):
    """Show reference image for the detected pose in a persistent window"""
    if pose_name and pose_name in reference_images:
        # Show clean image without text overlay
        img = reference_images[pose_name].copy()
        # Resize image to a reasonable size if it's too large or small
        height, width = img.shape[:2]
        if width > 400 or height > 400:
            scale = min(400 / width, 400 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        elif width < 200 or height < 200:
            scale = max(200 / width, 200 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        cv2.imshow(window_name, img)
    else:
        # Show a default/blank image with message if pose not found
        blank_img = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Pose Detected", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(blank_img, "Waiting for pose...", (40, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow(window_name, blank_img)

def main():
    """Main function to run the holistic detector with pose classification"""
    # Initialize detector and classifier
    detector = HolisticDetector()
    classifier = PoseClassifier()
    
    # Load reference images
    print("Loading reference images...")
    reference_images = load_reference_images()
    
    # Initialize webcam - try multiple camera indices
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Webcam opened successfully on camera index {camera_index}")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam")
        print("\nTroubleshooting steps:")
        print("1. Grant camera permissions:")
        print("   - Go to System Settings > Privacy & Security > Camera")
        print("   - Enable access for Terminal (or your IDE/editor)")
        print("2. Check if another app is using the webcam")
        print("3. Try disconnecting and reconnecting your webcam")
        return
    
    # Initialize reference image window
    reference_window_name = "Reference Image"
    
    print("MediaPipe Holistic Detector with Pose Classification Started!")
    print("Press 'q' to quit, 's' to save screenshot, 't' to train model")
    print("Reference image window will update with detected poses.")
    print(f"Look for a window titled '{reference_window_name}' - it may be behind the main window.")
    cv2.namedWindow(reference_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(reference_window_name, 400, 400)
    # Position window to the right of main window (adjust as needed)
    cv2.moveWindow(reference_window_name, 800, 100)
    
    # Show initial blank reference image with message
    blank_img = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.putText(blank_img, "Waiting for pose...", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(reference_window_name, blank_img)
    
    # Variables for sound timing
    last_sound_time = 0
    sound_cooldown = 0.5  # Minimum time between sounds in seconds
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect landmarks
        results = detector.detect(frame)
        
        # Draw landmarks
        frame = detector.draw_landmarks(frame, results)
        
        # Get landmark data and classify pose
        landmark_data = detector.get_landmark_data(results)
        pose_landmarks = landmark_data.get('pose')
        
        # Classify pose if landmarks are detected
        pose_prediction = "No Pose"
        confidence = 0.0
        all_confidences = {}
        if pose_landmarks is not None:
            pose_prediction, confidence = classifier.predict(pose_landmarks)
            # Get confidence for all classes
            all_confidences = classifier.get_all_confidences(pose_landmarks)
        
        # Show current pose in the reference window (always update to keep window visible)
        if pose_prediction != "No Pose" and pose_prediction in reference_images:
            show_reference_image(pose_prediction, reference_images, reference_window_name)
        else:  # Show "no pose" message if no pose landmarks detected
            show_reference_image(None, reference_images, reference_window_name)
        
        # Play sound with cooldown to avoid too frequent sounds
        current_time = time.time()
        if (pose_prediction != "No Pose" and 
            current_time - last_sound_time >= sound_cooldown):
            play_pose_sound(pose_prediction)
            last_sound_time = current_time
        
        # Add info text
        cv2.putText(frame, "MediaPipe Holistic Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show pose prediction with color coding based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        cv2.putText(frame, f"Pose: {pose_prediction} ({confidence:.2f})", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show all confidences for debugging
        if all_confidences:
            y_offset = 100
            for i, (pose_name, conf) in enumerate(all_confidences.items()):
                if conf > 0.1:  # Only show confidences above 10%
                    cv2.putText(frame, f"{pose_name}: {conf:.2f}", (10, y_offset + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to train", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('MediaPipe Holistic Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            cv2.imwrite('holistic_detection_screenshot.png', frame)
            print("Screenshot saved as 'holistic_detection_screenshot.png'")
        elif key == ord('t'):
            # Train model with collected data
            from data_collector import PoseDataCollector
            collector = PoseDataCollector()
            X, y = collector.load_sample_data()
            if X is not None:
                print("Model retrained with real data!")
            else:
                print("No data to train with. Collect some data first!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("Detection stopped successfully!")

if __name__ == "__main__":
    main()