# main.py
import sys
import cv2
import csv
import numpy as np
import glob
import argparse
import torch
from utilities import four_point_transform
from load_model import load_pytorch_model
from parking_model import ParkingSpaceModelS, ParkingSpaceModelM, classify_image_with_pytorch


def main(argv):
    parser = argparse.ArgumentParser(description="Parking Space Classification Comparison tool")
    parser.add_argument("--summary", help="Print only summary", action="store_true")
    args = parser.parse_args()

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

    # Load classifiers
    # TODO
    classifiers = [
        load_pytorch_model(ParkingSpaceModelS(), 'models/ParkingSpaceClassifierSmall.pth'),
        load_pytorch_model(ParkingSpaceModelM(), 'models/ParkingSpaceClassifierMedium.pth')
    ]

    # Load and process each test image
    test_images = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")
    test_images.sort()
    test_labels = sorted(glob.glob("test_images/*.txt"))
    score_sum = 0

    # For each parking lot image
    for img_idx, img_name in enumerate(test_images):
        image = cv2.imread(img_name)

        if image is None:
            print(f"Failed to load image {img_name}")
            continue

        # Read corresponding label file
        label_file = test_labels[img_idx]
        try:
            with open(label_file, 'r') as lf:
                true_labels = [int(line.strip()) for line in lf.readlines()]
        except FileNotFoundError:
            print(f"Label file {label_file} not found. Skipping image {img_name}.")
            continue

        if len(true_labels) != len(pkm_coordinates):
            print(f"Mismatch in number of labels and parking coordinates for {img_name}. Skipping.")
            continue

        annotated_image = image.copy()
        correct_predictions = 0
        total_predictions = len(pkm_coordinates)



        # For each parking spot image extracted from the parking lot image
        for idx, coord in enumerate(pkm_coordinates):
            points = [(coord[i], coord[i + 1]) for i in range(0, 8, 2)]
            one_place_img = four_point_transform(image, points)


            # TODO is_occupied = ??
            is_occupied = [
                classify_image_with_pytorch(classifiers[0], one_place_img),
                classify_image_with_pytorch(classifiers[1], one_place_img)
            ]

            # Compare prediction with ground truth
            true_label = true_labels[idx]
            is_correct = [False, False]
            for i in range(len(is_occupied)):
                is_correct[i] = (is_occupied[i] == (true_label == 1))

            text = f"S:{is_correct[0]} M:{is_correct[1]}"
            cv2.putText(annotated_image, text, (points[0][0], points[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Calculate and annotate success rate
        success_rate = (correct_predictions / total_predictions) * 100
        cv2.putText(annotated_image, f"Success Rate: {success_rate:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"Model on image {img_name} had success rate: {success_rate:.2f}%")
        score_sum += success_rate

        # Show the annotated image
        if (not args.summary):
            cv2.imshow("Annotated Parking Lot", annotated_image)
            if cv2.waitKey(0) & 0xFF == 27:  # Escape key
                break

    cv2.destroyAllWindows()
    print(f"Overall score: {score_sum / len(test_images)}")


if __name__ == "__main__":
    main(sys.argv[1:])
