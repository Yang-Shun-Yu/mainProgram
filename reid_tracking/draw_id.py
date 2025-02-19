import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Directories containing images and labels
image_dir = "/home/eddy/Desktop/train/test/images"
label_dir = "/home/eddy/Desktop/train/test/labels"
label_dir = '/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/reverse_labels'
output_dir = "/home/eddy/Desktop/train/test/reverse_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define a color palette (BGR format for OpenCV)
color_palette = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
]

# Function to get a color from the palette based on track_id
def get_color(track_id):
    return color_palette[track_id % len(color_palette)]

# Function to draw bounding boxes on an image using only labels
def draw_bounding_boxes(image_path, label_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None

    image_height, image_width = image.shape[:2]

    # Read the label file
    try:
        with open(label_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
        return None

    for line in lines:
        try:
            # Parse the annotation line (assuming YOLO format)
            # class_id, center_x, center_y, width, height = map(float, line.strip().split()[:5])
            class_id, center_x, center_y, width, height, id = map(float, line.split())
            new_track_id = str(int(id))
            # Convert normalized coordinates to pixel values
            center_x *= image_width
            center_y *= image_height
            width *= image_width
            height *= image_height

            # Calculate the top-left and bottom-right coordinates
            top_left_x = int(center_x - (width / 2))
            top_left_y = int(center_y - (height / 2))
            bottom_right_x = int(center_x + (width / 2))
            bottom_right_y = int(center_y + (height / 2))

            # Assign a random color based on class_id
            color = get_color(int(new_track_id))

            # Draw bounding box
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)

            # Add class_id as text label
            label = f'{new_track_id}'
            # label = f"Class {int(class_id)}"
            label_position = (top_left_x, top_left_y - 10 if top_left_y - 10 > 10 else top_left_y + 10)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        except ValueError:
            print(f"Error parsing line in {label_path}: {line}")
            continue

    return image

# Function to process an image-label pair
def process_image_label_pair(subdir, filename):
    image_path = os.path.join(image_dir, subdir, filename)
    label_path = os.path.join(label_dir, subdir, filename.replace(".jpg", ".txt"))
    output_subdir = os.path.join(output_dir, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, filename)

    if os.path.exists(label_path):
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image_path, label_path)
        if image_with_boxes is not None:
            # Save the annotated image
            cv2.imwrite(output_path, image_with_boxes)
    else:
        print(f"Label file not found for image: {image_path}")

# Gather all image-label pairs
image_label_pairs = []
for subdir in os.listdir(image_dir):
    sub_image_dir = os.path.join(image_dir, subdir)
    if os.path.isdir(sub_image_dir):
        for filename in os.listdir(sub_image_dir):
            if filename.endswith(".jpg"):
                image_label_pairs.append((subdir, filename))

# Process images in parallel with a progress bar
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_image_label_pair, subdir, filename): (subdir, filename) for subdir, filename in image_label_pairs}
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        pass
