from ultralytics import YOLO

class YoloModel():
    def __init__(self):
        self.model = None
        self.results = None

    def load_model(self):
        self.model = YOLO("yolo11l.pt")

    def process_image(self, image):
        self.results = self.model(image , verbose=False)

    def get_prediction(self, index):
        """
        Get prediction details for a specific detection index.
        Returns bounding box, class name, and confidence for the given index.
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results available. Process an image first.")

        # Get the detection results for the first image in the batch
        detections = self.results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = self.results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = self.results[0].boxes.cls.cpu().numpy()  # Class IDs

        if index < 0 or index >= len(detections):
            raise IndexError(f"Index {index} out of range for predictions.")

        # Extract the required prediction
        box = detections[index]
        confidence = confidences[index]
        class_id = int(class_ids[index])
        class_name = self.results[0].names[class_id]

        return {
            "bounding_box": box.tolist(),
            "class_name": class_name,
            "confidence": confidence
        }
    
    def get_top_predictions(self):
        """
        Get the top 5 predictions sorted by confidence.
        Returns bounding boxes, masks, class names, and confidence percentages.
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results available. Process an image first.")

        # Get the detection results for the first image in the batch
        detections = self.results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = self.results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = self.results[0].boxes.cls.cpu().numpy()  # Class IDs

        # Combine results and sort by confidence
        predictions = [
            {
                "bounding_box": detections[i].tolist(),
                "class_name": self.results[0].names[int(class_ids[i])],
                "confidence": confidences[i]
            }
            for i in range(len(detections))
        ]
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

        # Return the top 5 predictions
        return predictions

        

