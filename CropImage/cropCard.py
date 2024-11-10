import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

class CardProcessor:
    def __init__(self, image_path, debug=False):
        self.image_path = image_path
        self.debug = debug
        self.image = self._load_image()
        if self.image is None:
            raise FileNotFoundError(f"Error: Could not load image at {image_path}")

    def _load_image(self):
        """Load the image from the specified path."""
        return cv2.imread(self.image_path)

    def _preprocess_image(self):
        """Convert image to grayscale and apply a threshold."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def _find_card_contour(self, thresh):
        """Find the largest contour, assuming it is the card."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Error: No contours found.")
        return max(contours, key=cv2.contourArea)

    def _approximate_corners(self, contour):
        """Approximate contour to a polygon and ensure it has exactly 4 corners."""
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx_corners) != 4:
            raise ValueError(f"Error: Did not find exactly 4 corners. Found: {len(approx_corners)}")
        return approx_corners.reshape(4, 2)

    def _order_points(self, pts):
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _warp_card(self, src_points):
        """Warp perspective to get a top-down view of the card."""
        width, height = 1000, int((1000 * 3.25) / 2.5)
        dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(self.image, matrix, (width, height))

    def _extract_text_region(self, card):
        """Extract the region of interest containing the text."""
        height, width, _ = card.shape
        x_start_ratio, y_start_ratio = 0, 0.96
        width_ratio, height_ratio = 0.3, 0.04
        x = int(width * x_start_ratio)
        y = int(height * y_start_ratio)
        w = int(width * width_ratio)
        h = int(height * height_ratio)
        return card[y:y+h, x:x+w]
    
    def getProcessedCardImage(self):
        thresh = self._preprocess_image()
        card_contour = self._find_card_contour(thresh)
        approx_corners = self._approximate_corners(card_contour)
        src_points = self._order_points(approx_corners)
        card = self._warp_card(src_points)
        cropped_text_region = self._extract_text_region(card)
        return cv2.cvtColor(cropped_text_region, cv2.COLOR_BGR2RGB)

def process_card_image(image_path, debug=False):
    """Standalone function to process the card image and extract the text region."""
    processor = CardProcessor(image_path, debug=debug)
    return processor.getProcessedCardImage()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug = True

    try:
        cropped_text_region = process_card_image(image_path, debug=debug)
        if cropped_text_region is not None:
            plt.imshow(cropped_text_region)
            plt.title("Cropped Text Region")
            plt.show()
    except (FileNotFoundError, ValueError) as e:
        print(e)
