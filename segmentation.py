!git clone https://github.com/facebookresearch/segment-anything.git
%cd segment-anything
!pip install -e .

from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from google.colab import files

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load the SAM model
sam_checkpoint = "/content/drive/My Drive/sam_vit_h_4b8939.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
print("Model loaded successfully!")

# approach-1 grey scale


def calculate_disease_area_percentage(image_path, disease_threshold=0.5):
    # Load the image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Convert the image to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Thresholding or segmentation for isolating the leaf (example: leaf green color)
    leaf_mask = cv2.inRange(image_bgr, (20, 40, 20), (120, 200, 120))  # Adjust based on leaf color range

    # Convert leaf_mask to uint8 type to work with bitwise_and
    leaf_mask = leaf_mask.astype(np.uint8)

    # Apply the leaf mask on the image
    leaf_area = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)

    # Calculate the area of the leaf (non-zero pixels in the mask)
    leaf_area_pixels = np.sum(leaf_mask > 0)

    # Calculate the diseased area within the leaf area using a threshold
    # For simplicity, assuming the disease is detected by some thresholding
    diseased_area_mask = cv2.inRange(leaf_area, (50, 0, 0), (255, 100, 100))  # Adjust based on disease color
    diseased_area_mask = diseased_area_mask.astype(np.uint8)

    # Calculate the diseased area pixels
    diseased_area_pixels = np.sum(diseased_area_mask > 0)

    # Calculate the percentage of the leaf area that is diseased
    diseased_percentage = (diseased_area_pixels / leaf_area_pixels) * 100

    # Create the result image to show the mask
    result_img = cv2.bitwise_and(image_bgr, image_bgr, mask=diseased_area_mask)

    return diseased_percentage, diseased_area_mask, result_img

# Test the function with an uploaded image
uploaded_image = files.upload()
image_path = list(uploaded_image.keys())[0]
diseased_percentage, mask, result_img = calculate_disease_area_percentage(image_path)

# Plot the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(mask, cmap='gray')
plt.title('Disease Mask')
plt.subplot(1, 3, 2)
plt.imshow(result_img)
plt.title('Result Image')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f'Diseased Area: {diseased_percentage:.2f}%')
plt.show()


# approach -2 Segement anything Model
from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from google.colab import files

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load the SAM model
sam_checkpoint = "/content/drive/My Drive/sam_vit_h_4b8939.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
print("Model loaded successfully!")

def preprocess_image(image_np):
    # Convert to LAB color space to reduce lighting and shadow variations
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge CLAHE enhanced L channel back with A and B channels
    lab_image = cv2.merge((l, a, b))

    # Convert back to RGB
    image_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return image_rgb

def calculate_disease_area_percentage(image_path, disease_threshold=0.5):
    # Load the image and apply pre-processing
    image = Image.open(image_path)
    image_np = np.array(image)
    preprocessed_image = preprocess_image(image_np)

    # Convert the preprocessed image to BGR
    image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR)

    # Create a mask for the leaf area by defining color ranges for green
    leaf_mask = cv2.inRange(image_bgr, (20, 40, 20), (120, 200, 120))
    leaf_mask = leaf_mask.astype(np.uint8)

    # Apply the mask to isolate the leaf
    leaf_area = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)

    # Calculate the total area of the leaf in pixels
    leaf_area_pixels = np.sum(leaf_mask > 0)

    # Create a mask for diseased spots based on a different color range
    diseased_area_mask = cv2.inRange(leaf_area, (50, 0, 0), (255, 100, 100))
    diseased_area_mask = diseased_area_mask.astype(np.uint8)

    # Use morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    diseased_area_mask = cv2.morphologyEx(diseased_area_mask, cv2.MORPH_CLOSE, kernel)

    # Calculate diseased area in pixels
    diseased_area_pixels = np.sum(diseased_area_mask > 0)

    # Calculate the percentage of diseased area
    diseased_percentage = (diseased_area_pixels / leaf_area_pixels) * 100 if leaf_area_pixels > 0 else 0

    # Draw bounding boxes around each diseased spot
    contours, _ = cv2.findContours(diseased_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = image_bgr.copy()

    # Set a threshold for minimum contour area to filter out noise
    min_contour_area = 50  # Adjust based on desired precision

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box

    # Convert image back to RGB for matplotlib display
    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Display the final result with bounding boxes
    plt.figure(figsize=(5, 5))
    plt.imshow(output_img_rgb)
    plt.title(f"Result Image - Diseased Area: {diseased_percentage:.2f}%")
    plt.axis('off')
    plt.show()

    return diseased_percentage, output_img_rgb

# Test the function with an uploaded image
uploaded_image = files.upload()
image_path = list(uploaded_image.keys())[0]
diseased_percentage, result_img = calculate_disease_area_percentage(image_path)

# Print the result
print(f"Diseased Area Percentage: {diseased_percentage:.2f}%")
