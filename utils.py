# utils.py

import numpy as np
import cv2

def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts (list or array): List of four points (x, y).
    
    Returns:
        np.array: Ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and diff to find top-left and bottom-right
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def four_point_transform(image, pts):
    """
    Performs a perspective transform to obtain a top-down view of the image.
    
    Args:
        image (np.array): Input image.
        pts (list or array): List of four points (x, y).
    
    Returns:
        np.array: Warped image.
    """
    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect
    
    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Warp the image
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def extract_haar_feature_image(image, feature_size=(32, 32)):
    """
    Extracts Haar-like features from a grayscale image.
    
    Args:
        image (np.array): Grayscale input image.
        feature_size (tuple): Size to which the image is resized (width, height).
    
    Returns:
        list: List of Haar-like feature values.
    """
    # Resize image to a fixed size
    img = cv2.resize(image, feature_size)
    
    # Compute the integral image
    integral = cv2.integral(img)
    
    # Define Haar-like features
    # Each feature is defined by a type and its position and size
    # Types: 'two_horizontal', 'two_vertical', 'three_horizontal', 'three_vertical', 'four'
    features = []
    img_width, img_height = feature_size
    
    # Define feature size variations
    step_size = 4  # Step size for moving the window
    feature_width = 12
    feature_height = 12
    
    # Adjust feature dimensions if feature_size is increased
    # For example, with feature_size=(32,32), feature_width=16, feature_height=16
    # This ensures features scale with the image size
    # Uncomment the following lines if you increase feature_size
    # feature_width = 16
    # feature_height = 16
    
    # Two-rectangle features (edge features)
    for y in range(0, img_height - feature_height + 1, step_size):
        for x in range(0, img_width - 2 * feature_width + 1, step_size):
            # Horizontal two-rectangle feature
            white = sum_region(integral, x, y, feature_width, feature_height)
            black = sum_region(integral, x + feature_width, y, feature_width, feature_height)
            feature = white - black
            features.append(feature)
            
            # Vertical two-rectangle feature
            white = sum_region(integral, x, y, feature_width, feature_height)
            black = sum_region(integral, x, y + feature_height, feature_width, feature_height)
            feature = white - black
            features.append(feature)
    
    # Three-rectangle features (line features)
    for y in range(0, img_height - feature_height + 1, step_size):
        for x in range(0, img_width - 3 * feature_width + 1, step_size):
            # Horizontal three-rectangle feature
            white = sum_region(integral, x, y, feature_width, feature_height)
            black = sum_region(integral, x + feature_width, y, feature_width, feature_height)
            white2 = sum_region(integral, x + 2 * feature_width, y, feature_width, feature_height)
            feature = white - black + white2
            features.append(feature)
    
    for y in range(0, img_height - 3 * feature_height + 1, step_size):
        for x in range(0, img_width - feature_width + 1, step_size):
            # Vertical three-rectangle feature
            white = sum_region(integral, x, y, feature_width, feature_height)
            black = sum_region(integral, x, y + feature_height, feature_width, feature_height)
            white2 = sum_region(integral, x, y + 2 * feature_height, feature_width, feature_height)
            feature = white - black + white2
            features.append(feature)
    
    # Four-rectangle features
    for y in range(0, img_height - 2 * feature_height + 1, step_size):
        for x in range(0, img_width - 2 * feature_width + 1, step_size):
            white = sum_region(integral, x, y, feature_width, feature_height)
            black = sum_region(integral, x + feature_width, y, feature_width, feature_height)
            white2 = sum_region(integral, x, y + feature_height, feature_width, feature_height)
            black2 = sum_region(integral, x + feature_width, y + feature_height, feature_width, feature_height)
            feature = white - black - white2 + black2
            features.append(feature)
    
    return features

def sum_region(integral, x, y, w, h):
    """
    Computes the sum of pixel intensities within a rectangular region using the integral image.
    
    Args:
        integral (np.array): Integral image.
        x (int): Top-left x-coordinate.
        y (int): Top-left y-coordinate.
        w (int): Width of the rectangle.
        h (int): Height of the rectangle.
    
    Returns:
        int: Sum of pixel intensities within the region.
    """
    # Ensure that the region does not exceed the integral image boundaries
    max_y, max_x = integral.shape
    y_end = y + h
    x_end = x + w
    
    if y_end >= max_y:
        y_end = max_y - 1
    if x_end >= max_x:
        x_end = max_x - 1
    
    A = integral[y, x]
    B = integral[y, x_end]
    C = integral[y_end, x]
    D = integral[y_end, x_end]
    return D - B - C + A
