# main.py
import sys
import cv2
import csv
import numpy as np
import glob
import argparse
import torch
from yolo import YoloModel
from utilities import four_point_transform, extract_haar_feature_image
from nn import load_training_data, load_classifier, train_classifier
from edge_classifier import get_number_of_edge_pixels
from parking_model import classify_image_with_pytorch, ParkingSpaceClassifier
from load_model import load_pytorch_model
from googlenet_model import GoogLeNetSmall, classify_image_with_googlenet
from resnet_model import ResNet18, classify_image_with_resnet18
from shapely.geometry import Polygon

accepted_classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'boat' ]


def main(argv):
    parser = argparse.ArgumentParser(description="Parking Space Classification Tool")
    parser.add_argument("model", choices=["edge", "haar", "pytorch", "googlenet", "resnet"],
                    help="Classification model to use")
    parser.add_argument("--segment", help="Use segmentation and object detection to help classification", action="store_true")
    parser.add_argument("--summary", help="Print only summary", action="store_true")
    args = parser.parse_args()
    

    # create results csv file
    results_csv = f"results/results-{args.model}.csv"
    results_csv_header = ["image", "success_rate"]

    # Write header and rows to the CSV file
    with open(results_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(results_csv_header)  # Write the header

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

    # Load classifier 
    if args.model == 'haar':
        model_path = 'models/random_forest_model.pkl'
        clf = load_classifier(model_path)
        if clf is None:
            X, y = load_training_data()
            clf = train_classifier(X, y, model_path=model_path)
    elif args.model == 'pytorch':
        pytorch_model = load_pytorch_model(ParkingSpaceClassifier(), 'models/parking_space_cnn.pth')
    elif args.model == 'googlenet':
        googlenet_model = load_pytorch_model(GoogLeNetSmall(), 'models/googlenet_model.pth')
    elif args.model == 'resnet':
        resnet_model = load_pytorch_model(ResNet18(), 'models/resnet18_model.pth');

    if args.segment:
        yolo = YoloModel()
        yolo.load_model()

    # Load and process each test image
    test_images = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")
    test_images.sort()
    test_labels = sorted(glob.glob("test_images/*.txt"))
    rates_sum = 0

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

        if args.segment:
            yolo.process_image(image)
            yolo_predictions = yolo.get_top_predictions()

        # For each parking spot image extracted from the parking lot image
        for idx, coord in enumerate(pkm_coordinates):
            points = [(coord[i], coord[i + 1]) for i in range(0, 8, 2)]
            one_place_img = four_point_transform(image, points)
            parking_polygon = Polygon(points)

            # Filter YOLO predictions based on IoU
            overlapping_predictions = []
            for p in yolo_predictions:
                bbox = p['bounding_box']  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = map(int, bbox)
                bbox_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                if bbox_polygon.intersects(parking_polygon):
                    intersection_area = bbox_polygon.intersection(parking_polygon).area
                    union_area = bbox_polygon.union(parking_polygon).area
                    iou = intersection_area / union_area
                    if iou > 0.1:  # Significant overlap threshold
                        overlapping_predictions.append((p, iou))

            if args.model == 'edge':
                edge_count = get_number_of_edge_pixels(one_place_img)
                is_occupied = edge_count >= 500  # threshold
            elif args.model == 'haar':
                features = extract_haar_feature_image(cv2.cvtColor(one_place_img, cv2.COLOR_BGR2GRAY))
                is_occupied = clf.predict([features])[0] == 1 if features else False
            elif args.model == 'pytorch':
                is_occupied = classify_image_with_pytorch(pytorch_model, one_place_img)
            elif args.model == 'googlenet':
                is_occupied = classify_image_with_googlenet(googlenet_model, one_place_img)
            elif args.model == 'resnet':
                is_occupied = classify_image_with_resnet18(resnet_model, one_place_img)
            
             # Select the prediction with the highest confidence
            if overlapping_predictions:
                best_prediction, best_iou = max(overlapping_predictions, key=lambda x: x[0]['confidence'])

                # Draw the best prediction
                bbox = best_prediction['bounding_box']
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                label = f"{best_prediction['class_name']} {best_prediction['confidence']:.2f}, IoU: {best_iou:.2f}"
                cv2.putText(annotated_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Update `is_occupied` based on the best prediction's IoU
                iou = max([iou for _, iou in overlapping_predictions])
                if iou > 0.3 and is_occupied == 0:
                    is_occupied = 1
                elif iou < 0.1 and is_occupied == 1:
                    is_occupied = 0

                # 0.3 0.1

            # Compare prediction with ground truth
            true_label = true_labels[idx]
            is_correct = (is_occupied == (true_label == 1))

            # Annotate the parking spot
            if is_correct:
                color = (0, 0, 255) if is_occupied else (0, 255, 0)  # Red for full, Green for free
                correct_predictions += 1
            else:
                color = (255, 0, 255)  # Purple for incorrect prediction

            cv2.polylines(annotated_image, [np.array(points, np.int32).reshape((-1, 1, 2))], True, color, 2)
            if not is_correct:
                text = "Incorrect full" if is_occupied == 1 else "Incorrect free"
                cv2.putText(annotated_image, text, (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Calculate and annotate success rate
        success_rate = (correct_predictions / total_predictions) * 100
        cv2.putText(annotated_image, f"Success Rate: {success_rate:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"{args.model} model on image {img_name} had success rate: {success_rate:.2f}%")
        rates_sum += success_rate
        with open(results_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([img_name, success_rate])  # Write the header


        # Show the annotated image
        if(not args.summary):
            cv2.imshow("Annotated Parking Lot", annotated_image)
            if cv2.waitKey(0) & 0xFF == 27:  # Escape key
                break

    cv2.destroyAllWindows()
    print(f"Average success rate: {rates_sum / len(test_images)}")

if __name__ == "__main__":
    main(sys.argv[1:])
