#!/usr/bin/env python3
"""
Script to inspect .pkl and .npy files in the project
"""

import pickle
import numpy as np
import json
import os

def inspect_pkl_file(filepath):
    """Inspect a pickle file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        print(f"\nType: {type(obj)}")
        print(f"Size: {os.path.getsize(filepath)} bytes")
        
        # If it's a scikit-learn model
        if hasattr(obj, '__class__'):
            print(f"\nClass: {obj.__class__.__name__}")
            print(f"Module: {obj.__class__.__module__}")
            
            # Try to get model attributes
            if hasattr(obj, 'n_estimators'):
                print(f"\nModel Type: RandomForestClassifier")
                print(f"Number of estimators: {obj.n_estimators}")
                print(f"Max depth: {obj.max_depth}")
                print(f"Random state: {obj.random_state}")
            
            if hasattr(obj, 'classes_'):
                print(f"Classes: {obj.classes_}")
                print(f"Number of classes: {len(obj.classes_)}")
            
            if hasattr(obj, 'n_features_in_'):
                print(f"Number of input features: {obj.n_features_in_}")
        
        print(f"\nObject representation (first 500 chars):")
        print(str(obj)[:500])
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def inspect_npy_file(filepath):
    """Inspect a numpy array file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        arr = np.load(filepath, allow_pickle=True)
        
        print(f"\nType: {type(arr)}")
        print(f"Shape: {arr.shape}")
        print(f"Size: {arr.size} elements")
        print(f"Data type: {arr.dtype}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        
        if arr.size > 0:
            print(f"\nMin value: {np.min(arr)}")
            print(f"Max value: {np.max(arr)}")
            print(f"Mean value: {np.mean(arr):.4f}")
            print(f"Std deviation: {np.std(arr):.4f}")
            
            print(f"\nFirst few values:")
            if arr.ndim == 1:
                print(arr[:10])
            elif arr.ndim == 2:
                print(arr[:5, :10])
            else:
                print(arr.flat[:10])
        
    except Exception as e:
        print(f"Error loading numpy file: {e}")

def inspect_json_file(filepath):
    """Inspect a JSON file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\nType: {type(data)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        print(f"\nContent:")
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        print(f"Error loading JSON file: {e}")

if __name__ == "__main__":
    print("RoyalReactions - File Inspector")
    print("=" * 60)
    
    # Inspect model file
    model_file = "pose_classifier_model.pkl"
    if os.path.exists(model_file):
        inspect_pkl_file(model_file)
    else:
        print(f"\n{model_file} not found")
    
    # Inspect training data files
    data_dir = "pose_data"
    if os.path.exists(data_dir):
        features_file = os.path.join(data_dir, "pose_features_latest.npy")
        labels_file = os.path.join(data_dir, "pose_labels_latest.npy")
        metadata_file = os.path.join(data_dir, "pose_metadata_latest.json")
        
        if os.path.exists(features_file):
            inspect_npy_file(features_file)
        
        if os.path.exists(labels_file):
            inspect_npy_file(labels_file)
        
        if os.path.exists(metadata_file):
            inspect_json_file(metadata_file)
    else:
        print(f"\n{data_dir} directory not found")
    
    print(f"\n{'='*60}")
    print("Inspection complete!")
    print(f"{'='*60}")

