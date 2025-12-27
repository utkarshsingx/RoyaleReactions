import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier

class PoseDataCollector:
    def __init__(self, data_dir="pose_data"):
        """
        Initialize pose data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.detector = HolisticDetector()
        self.classifier = PoseClassifier()
        self.data_dir = data_dir
        self.current_pose = 0
        self.collected_samples = 0
        self.samples_per_pose = 100
        self.auto_collect = False
        self.collection_delay = 10
        self.frame_counter = 0
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Pose labels
        self.pose_labels = {
            0: "Laughing",
            1: "Yawning", 
            2: "Crying",
            3: "Taunting"
        }
        
        print("Pose Data Collector - Auto Collection Mode")
        print("Controls:")
        print("  '0' - Set pose to Laughing") # Adjust if you want to change the pose labels
        print("  '1' - Set pose to Yawning") 
        print("  '2' - Set pose to Crying")
        print("  '3' - Set pose to Taunting")
        print("  'a' - Toggle auto collection (start/stop)")
        print("  's' - Save collected data to files")
        print("  't' - Train model with collected data")
        print("  'l' - Load previously saved data")
        print("  'q' - Quit")
        print("\nAuto collection will automatically collect frames as samples!")
    
    def collect_data(self):
        """Main data collection loop"""
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        # Try different camera indices if 0 doesn't work
        if not cap.isOpened():
            print("Camera 0 not available, trying camera 1...")
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("Error: Could not open any webcam")
            print("Please check:")
            print("1. Webcam is connected")
            print("2. No other application is using the webcam")
            print("3. Webcam permissions are granted")
            return
        
        print("Webcam initialized successfully!")
        
        collected_data = []
        collected_labels = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.frame_counter += 1
            
            # Detect landmarks
            results = self.detector.detect(frame)
            
            # Draw landmarks
            frame = self.detector.draw_landmarks(frame, results)
            
            # Get pose data
            landmark_data = self.detector.get_landmark_data(results)
            pose_landmarks = landmark_data.get('pose')
            
            # Auto-collect samples if enabled
            if (self.auto_collect and 
                pose_landmarks is not None and 
                self.frame_counter % self.collection_delay == 0):
                
                # Extract features
                features = self.classifier.extract_features(pose_landmarks)
                collected_data.append(features)
                collected_labels.append(self.current_pose)
                self.collected_samples += 1
                print(f"Auto-collected sample {self.collected_samples} for {self.pose_labels[self.current_pose]}")
                
                # Check if we've collected enough samples for current pose
                if self.collected_samples >= self.samples_per_pose:
                    print(f"Completed collecting {self.samples_per_pose} samples for {self.pose_labels[self.current_pose]}")
                    print(f"Moving to next pose...")
                    self.collected_samples = 0
                    self.current_pose = (self.current_pose + 1) % len(self.pose_labels)
                    if self.current_pose == 0:
                        print("All poses completed! You can train the model now.")
                        self.auto_collect = False  # Stop auto-collection
            
            # Add instructions to frame
            auto_status = "ON" if self.auto_collect else "OFF"
            cv2.putText(frame, f"Current Pose: {self.pose_labels[self.current_pose]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples Collected: {self.collected_samples}/{self.samples_per_pose}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Auto Collection: {auto_status}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 0-3 to change pose, 'a' to toggle auto-collect", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "'s' to save, 'l' to load, 't' to train", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add visual indicator for auto-collection
            if self.auto_collect and pose_landmarks is not None:
                cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)  # Green circle when collecting
                cv2.putText(frame, "COLLECTING", (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif self.auto_collect:
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)  # Red circle when no pose detected
                cv2.putText(frame, "NO POSE", (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow('Pose Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('3'):
                self.current_pose = key - ord('0')
                self.collected_samples = 0
                print(f"Switched to collecting: {self.pose_labels[self.current_pose]}")
            elif key == ord('a'):
                self.auto_collect = not self.auto_collect
                status = "started" if self.auto_collect else "stopped"
                print(f"Auto collection {status}!")
                if self.auto_collect:
                    print(f"Now collecting samples for: {self.pose_labels[self.current_pose]}")
                    print("Hold your pose steady and the system will automatically collect samples.")
            elif key == ord('s'):
                if len(collected_data) > 0:
                    self.save_data(collected_data, collected_labels)
                else:
                    print("No data to save yet!")
            elif key == ord('l'):
                loaded_data, loaded_labels = self.load_data()
                if loaded_data is not None:
                    collected_data = loaded_data
                    collected_labels = loaded_labels
                    print(f"Loaded {len(collected_data)} samples from file!")
            elif key == ord('t'):
                if len(collected_data) > 0:
                    self.train_model(collected_data, collected_labels)
                else:
                    print("No data collected yet!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
    
    def train_model(self, X, y):
        """Train the model with collected data"""
        if len(X) < 10:
            print("Need at least 10 samples to train. Collect more data!")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training model with {len(X)} samples...")
        self.classifier.train_model(X, y)
        print("Model training completed!")
    
    def save_data(self, X, y):
        """Save collected data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save features and labels
        features_file = os.path.join(self.data_dir, f"pose_features_{timestamp}.npy")
        labels_file = os.path.join(self.data_dir, f"pose_labels_{timestamp}.npy")
        
        np.save(features_file, np.array(X))
        np.save(labels_file, np.array(y))
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "num_samples": len(X),
            "pose_labels": self.pose_labels,
            "samples_per_pose": self.samples_per_pose,
            "collection_date": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.data_dir, f"pose_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data saved to {self.data_dir}/")
        print(f"  Features: pose_features_{timestamp}.npy ({len(X)} samples)")
        print(f"  Labels: pose_labels_{timestamp}.npy")
        print(f"  Metadata: pose_metadata_{timestamp}.json")
        
        # Also save to a default file for easy loading
        np.save(os.path.join(self.data_dir, "pose_features_latest.npy"), np.array(X))
        np.save(os.path.join(self.data_dir, "pose_labels_latest.npy"), np.array(y))
        with open(os.path.join(self.data_dir, "pose_metadata_latest.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Also saved as 'latest' files for easy loading!")
    
    def load_data(self, filename_prefix="latest"):
        """Load previously saved data"""
        features_file = os.path.join(self.data_dir, f"pose_features_{filename_prefix}.npy")
        labels_file = os.path.join(self.data_dir, f"pose_labels_{filename_prefix}.npy")
        
        if not os.path.exists(features_file) or not os.path.exists(labels_file):
            print(f"No saved data found with prefix '{filename_prefix}'")
            print("Available files:")
            for file in os.listdir(self.data_dir):
                if file.startswith("pose_features_"):
                    print(f"  {file}")
            return None, None
        
        try:
            X = np.load(features_file)
            y = np.load(labels_file)
            
            print(f"Loaded {len(X)} samples from {filename_prefix} files")
            return X.tolist(), y.tolist()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def load_sample_data(self):
        """
        Load previously saved real data for training
        No synthetic data - only real collected data
        """
        print("Loading real collected data...")
        
        # Try to load saved data
        X, y = self.load_data("latest")
        
        if X is None:
            print("No real data found!")
            print("Please collect some data first:")
            print("1. Run: python data_collector.py")
            print("2. Collect samples for each pose (press 0-3 to select pose, 'a' to auto-collect)")
            print("3. Save your data (press 's')")
            print("4. Then try training again")
            return None, None
        
        print(f"Found {len(X)} real samples")
        self.classifier.train_model(X, y)
        print("Model trained with real data!")
        
        return X, y

# Add a main function for direct execution
if __name__ == "__main__":
    collector = PoseDataCollector()
    collector.collect_data()
