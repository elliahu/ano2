# main.py
import sys
import cv2
import numpy as np
import glob
from utils import four_point_transform, extract_haar_feature_image
from nn import load_training_data, load_classifier, train_classifier
from edge_classifier import get_number_of_edge_pixels
from parking_model import classify_image_with_pytorch, ParkingSpaceClassifier
from load_model import load_pytorch_model
from googlenet_model import GoogLeNetSmall, classify_image_with_googlenet

def main(argv):
    # Allow user to select classification method
    if len(argv) > 0 and argv[0] in ('edge', 'haar', 'pytorch', 'googlenet'):
        method = argv[0]
    else:
        print("Usage: main.py <classification_method>")
        print("classification_method options: 'edge', 'haar', 'pytorch', 'googlenet")
        sys.exit(1)

    # Read parking space coordinates from file
    try:
        with open('parking_map_python.txt', 'r') as pkm_file:
            pkm_lines = pkm_file.readlines()
    except FileNotFoundError:
        print("Error: 'parking_map_python.txt' not found.")
        sys.exit(1)

    # Parse parking coordinates
    pkm_coordinates = []
    for line in pkm_lines:
        coord = list(map(int, line.strip().split()))
        if len(coord) == 8:
            pkm_coordinates.append(coord)

    # Load classifier if using Haar or PyTorch model
    if method == 'haar':
        model_path = 'models/random_forest_model.pkl'
        clf = load_classifier(model_path)
        if clf is None:
            X, y = load_training_data()
            clf = train_classifier(X, y, model_path=model_path)
    elif method == 'pytorch':
        pytorch_model = load_pytorch_model(ParkingSpaceClassifier(), 'models/parking_space_cnn.pth')
    elif method == 'googlenet':
        googlenet_model = load_pytorch_model(GoogLeNetSmall(), 'models/googlenet_model.pth')

    # Load and process each test image
    test_images = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")
    test_images.sort()
    for img_name in test_images:
        image = cv2.imread(img_name)
        if image is None:
            print(f"Failed to load image {img_name}")
            continue

        annotated_image = image.copy()
        for coord in pkm_coordinates:
            points = [(coord[i], coord[i+1]) for i in range(0, 8, 2)]
            one_place_img = four_point_transform(image, points)

            if method == 'edge':
                edge_count = get_number_of_edge_pixels(one_place_img)
                is_occupied = edge_count >= 500  # threshold
            elif method == 'haar':
                features = extract_haar_feature_image(cv2.cvtColor(one_place_img, cv2.COLOR_BGR2GRAY))
                if features:
                    is_occupied = clf.predict([features])[0] == 1
                else:
                    is_occupied = False
            elif method == 'pytorch':
                is_occupied = classify_image_with_pytorch(pytorch_model, one_place_img)
            elif method == 'googlenet':
                is_occupied = classify_image_with_googlenet(googlenet_model, one_place_img)
            

            # Annotate the image based on the result
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.polylines(annotated_image, [np.array(points, np.int32).reshape((-1, 1, 2))], True, color, 2)

        # Show the annotated image
        cv2.imshow("Annotated Parking Lot", annotated_image)
        if cv2.waitKey(0) & 0xFF == 27:  # Escape key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
