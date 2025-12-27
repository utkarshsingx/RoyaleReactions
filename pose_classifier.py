import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class PoseClassifier:
    def __init__(self, model_path=None):
        """
        Initialize pose classifier
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.pose_labels = {
            0: "Laughing",
            1: "Yawning", 
            2: "Crying",
            3: "Taunting",
            4: "Unknown"
        }
        self.model_path = model_path or "pose_classifier_model.pkl"
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            self.load_model()
    
    def extract_features(self, pose_landmarks):
        """
        Extract features from pose landmarks for classification
        
        Args:
            pose_landmarks: Array of pose landmarks (33, 3)
            
        Returns:
            features: Extracted feature vector
        """
        if pose_landmarks is None:
            return np.zeros(20)  # Return zeros if no pose detected
        
        # Convert to numpy array if needed
        landmarks = np.array(pose_landmarks)
        
        # Extract key features for pose classification
        features = []
        
        # 1. Shoulder width (distance between shoulders)
        left_shoulder = landmarks[11]  # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)
        
        # 2. Hip width (distance between hips)
        left_hip = landmarks[23]  # Left hip
        right_hip = landmarks[24]  # Right hip
        hip_width = np.linalg.norm(left_hip - right_hip)
        features.append(hip_width)
        
        # 3. Body height (shoulder to hip distance)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        body_height = np.linalg.norm(shoulder_center - hip_center)
        features.append(body_height)
        
        # 4. Arm angles (elbow to shoulder to wrist angles)
        # Left arm
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_arm_angle = self._calculate_angle(left_elbow, left_shoulder, left_wrist)
        features.append(left_arm_angle)
        
        # Right arm
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        right_arm_angle = self._calculate_angle(right_elbow, right_shoulder, right_wrist)
        features.append(right_arm_angle)
        
        # 5. Hand heights relative to shoulders
        left_hand_height = left_shoulder[1] - left_wrist[1]  # Y coordinate difference
        right_hand_height = right_shoulder[1] - right_wrist[1]
        features.extend([left_hand_height, right_hand_height])
        
        # 6. Knee angles
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_knee_angle, right_knee_angle])
        
        # 7. Torso orientation (shoulder to hip angle)
        torso_angle = self._calculate_angle(left_shoulder, hip_center, right_hip)
        features.append(torso_angle)
        
        # 8. Arm extension (distance from shoulder to wrist)
        left_arm_extension = np.linalg.norm(left_shoulder - left_wrist)
        right_arm_extension = np.linalg.norm(right_shoulder - right_wrist)
        features.extend([left_arm_extension, right_arm_extension])
        
        # 9. Leg positions relative to hips
        left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_leg_angle, right_leg_angle])
        
        # 10. Overall body symmetry
        left_side_avg = np.mean([left_shoulder, left_elbow, left_wrist, left_hip, left_knee], axis=0)
        right_side_avg = np.mean([right_shoulder, right_elbow, right_wrist, right_hip, right_knee], axis=0)
        body_symmetry = np.linalg.norm(left_side_avg - right_side_avg)
        features.append(body_symmetry)
        
        # 11. Hand movement indicators (distance from center)
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_hand_offset = abs(left_wrist[0] - center_x)
        right_hand_offset = abs(right_wrist[0] - center_x)
        features.extend([left_hand_offset, right_hand_offset])
        
        # 12. Vertical body alignment
        nose = landmarks[0]
        vertical_alignment = abs(nose[0] - (left_shoulder[0] + right_shoulder[0]) / 2)
        features.append(vertical_alignment)
        
        return np.array(features)
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        
        Args:
            point1, point2, point3: Three 3D points
            
        Returns:
            angle: Angle in degrees
        """
        # Create vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def train_model(self, X, y):
        """
        Train the pose classification model
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        
        # Get unique labels in the data
        unique_labels = np.unique(y)
        target_names = [self.pose_labels.get(label, f"Class_{label}") for label in unique_labels]
        
        print(classification_report(y_test, y_pred, 
                                  labels=unique_labels,
                                  target_names=target_names))
        
        # Save model
        self.save_model()
    
    def predict(self, pose_landmarks):
        """
        Predict pose from landmarks
        
        Args:
            pose_landmarks: Pose landmarks array
            
        Returns:
            prediction: Predicted pose label
            confidence: Prediction confidence
        """
        if self.model is None:
            return "No Model", 0.0
        
        # Extract features
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        pose_name = self.pose_labels.get(prediction, "Unknown")
        
        return pose_name, confidence
    
    def get_all_confidences(self, pose_landmarks):
        """
        Get confidence scores for all pose classes
        
        Args:
            pose_landmarks: Pose landmarks array
            
        Returns:
            confidences: Dictionary of pose names and their confidence scores
        """
        if self.model is None:
            return {}
        
        # Extract features
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)
        
        # Get probabilities for all classes
        probabilities = self.model.predict_proba(features)[0]
        
        # Map to pose names
        confidences = {}
        for i, prob in enumerate(probabilities):
            pose_name = self.pose_labels.get(i, f"Class_{i}")
            confidences[pose_name] = prob
        
        return confidences
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a saved model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
