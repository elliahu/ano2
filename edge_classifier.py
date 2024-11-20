import cv2

threshold = 500

def get_number_of_edge_pixels(img, canny_p1 = 80, canny_p2 = 180) -> int:
    # Apply grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, canny_p1, canny_p2)

    # Count the number of edge pixels
    count = cv2.countNonZero(edges)

    return count
            

def is_space_occupied(img) -> bool:
    return get_number_of_edge_pixels(img) > threshold
