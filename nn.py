import cv2
import glob
import os
from utilities import extract_haar_feature_image
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving and loading the trained model



def load_training_data(train_full_dir='train_images/full', train_free_dir='train_images/free'):
    """
    Load training images, extract Haar-like features, and create labels.
    
    Args:
        train_full_dir (str): Directory containing images of occupied parking spaces.
        train_free_dir (str): Directory containing images of free parking spaces.
    
    Returns:
        X (list): Feature vectors.
        y (list): Labels (1 for occupied, 0 for free).
    """
    X = []
    y = []
    
    # Load occupied parking space images
    full_images = glob.glob(os.path.join(train_full_dir, "*.jpg")) + glob.glob(os.path.join(train_full_dir, "*.png"))
    print(f"Loading {len(full_images)} occupied parking space images from '{train_full_dir}'...")
    for img_path in full_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load image {img_path}. Skipping.")
            continue
        features = extract_haar_feature_image(img)
        if features:  # Ensure features are extracted
            X.append(features)
            y.append(1)  # 1 indicates occupied
    
    # Load free parking space images
    free_images = glob.glob(os.path.join(train_free_dir, "*.jpg")) + glob.glob(os.path.join(train_free_dir, "*.png"))
    print(f"Loading {len(free_images)} free parking space images from '{train_free_dir}'...")
    for img_path in free_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load image {img_path}. Skipping.")
            continue
        features = extract_haar_feature_image(img)
        if features:  # Ensure features are extracted
            X.append(features)
            y.append(0)  # 0 indicates free
    
    print(f"Total training samples: {len(X)}")
    return X, y


def train_classifier(X, y, model_path='random_forest_model.pkl'):
    """
    Train a Random Forest classifier and save the model.
    
    Args:
        X (list): Feature vectors.
        y (list): Labels.
        model_path (str): Path to save the trained model.
    
    Returns:
        clf (RandomForestClassifier): Trained classifier.
    """
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    print(f"Model trained and saved to '{model_path}'.")
    return clf

def load_classifier(model_path='random_forest_model.pkl'):
    """
    Load a pre-trained classifier from disk.
    
    Args:
        model_path (str): Path to the trained model.
    
    Returns:
        clf (RandomForestClassifier): Loaded classifier.
    """
    if os.path.exists(model_path):
        print(f"Loading trained classifier from '{model_path}'...")
        clf = joblib.load(model_path)
        return clf
    else:
        print(f"Model file '{model_path}' not found.")
        return None
