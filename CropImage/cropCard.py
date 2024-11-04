import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def order_points(pts):
    # Initialize a list of coordinates
    rect = np.zeros((4, 2), dtype="float32")

    # Sum the coordinates to determine the top-left and bottom-right points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point
    rect[2] = pts[np.argmax(s)]  # Bottom-right point

    # Difference to determine the top-right and bottom-left points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point

    return rect


def process_card_image(image_path, debug=False): 
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to keep only very dark regions (adjust the threshold as needed)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Detects dark regions

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest rectangular contour is the card
    card_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon and get the corner points
    epsilon = 0.1 * cv2.arcLength(card_contour, True)  # Approximation accuracy
    approx_corners = cv2.approxPolyDP(card_contour, epsilon, True)

    # Check if we found exactly 4 points (corners)
    if len(approx_corners) != 4:
        print("Error: Did not find exactly 4 corners. Found:", len(approx_corners))
        exit()

    # Order the points
    src_points = order_points(approx_corners.reshape(4, 2))

    # Define the destination points (where the corners should go)
    width = 1000  # Set the desired width
    height = int((width * 3.25) / 2.5)  # Calculate height based on the aspect ratio
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height] ], dtype='float32')

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp
    card = cv2.warpPerspective(image, matrix, (width, height))

    # Get the dimensions of the image
    height, width, _ = card.shape

    # Define the relative position and size of the cropping box
    x_start_ratio, y_start_ratio = 0, 0.96  # Starting point as a fraction of width/height
    width_ratio, height_ratio = 0.3, 0.04    # Width and height as fractions of the image size

    # Calculate absolute coordinates based on image dimensions
    x = int(width * x_start_ratio)
    y = int(height * y_start_ratio)
    w = int(width * width_ratio)
    h = int(height * height_ratio)

    # Crop the region of interest (ROI) containing the text
    cropped_text_region = card[y:y+h, x:x+w]
    cropped_text_region= cv2.cvtColor(cropped_text_region, cv2.COLOR_BGR2RGB)
    return cropped_text_region

if __name__ == "__main__":
    # Take the image path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = True  # Set debug mode to True for visualizing contours

    cropped_text_region = process_card_image(image_path, debug=debug)

    if cropped_text_region is not None:
        # Display the cropped text region
        plt.imshow(cv2.cvtColor(cropped_text_region, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Text Region")
        plt.show()
