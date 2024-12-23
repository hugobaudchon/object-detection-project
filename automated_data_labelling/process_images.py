import os

from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
from PIL import Image
from tqdm import tqdm
from PIL import ImageDraw

zip_file_images_folder = "duckietown_images"

class_list = [
    "rc car",
    "yellow rubber duckie",
    "small orange traffic cone",
    "wooden toy house",
    "qr code",
    "road sign"
]

DETECTION_THRESHOLD = 0.2
CONTAINMENT_THRESHOLD = 0.7

num_images = len(os.listdir(zip_file_images_folder))

print(f"Number of images: {num_images}")

print("Loading OWLv2 model...")

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda()

import numpy as np

def filter_redundant_boxes_owlv2(boxes, scores, labels, containment_threshold=CONTAINMENT_THRESHOLD):
    """
    Filter out larger bounding boxes that mostly contain multiple smaller boxes of the same class.
    Works with OWLv2 format boxes (x1,y1,x2,y2).
    """
    def calculate_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])  # (x2-x1) * (y2-y1)

    def calculate_intersection(box1, box2):
        """Calculate intersection area of two boxes in (x1,y1,x2,y2) format"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        return (x_right - x_left) * (y_bottom - y_top)

    def mostly_contains(large_box, small_box):
        """Check if large_box mostly contains small_box"""
        intersection = calculate_intersection(large_box, small_box)
        small_box_area = calculate_area(small_box)

        containment_ratio = intersection / small_box_area
        return containment_ratio > containment_threshold

    n_boxes = len(boxes)
    if n_boxes == 0:
        return np.array([], dtype=int)

    keep_mask = np.ones(n_boxes, dtype=bool)
    areas = np.array([calculate_area(box) for box in boxes])
    sorted_indices = np.argsort(-areas)  # Sort by area, largest first

    for i in range(n_boxes):
        if not keep_mask[sorted_indices[i]]:
            continue

        current_box = boxes[sorted_indices[i]]
        current_label = labels[sorted_indices[i]]

        # Look for smaller boxes that this box mostly contains
        contained_boxes = []
        for j in range(i + 1, n_boxes):
            if not keep_mask[sorted_indices[j]]:
                continue

            compare_box = boxes[sorted_indices[j]]
            compare_label = labels[sorted_indices[j]]

            if compare_label == current_label and mostly_contains(current_box, compare_box):
                contained_boxes.append(sorted_indices[j])

        # If this box mostly contains multiple smaller boxes of the same class, remove it
        if len(contained_boxes) >= 2:
            keep_mask[sorted_indices[i]] = False

    return np.where(keep_mask)[0]

def process_image_folder(folder_path, processor, model, texts, batch_size=16):
    """Process all images in a folder through OWLv2 model and filter redundant boxes"""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []

    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        images = [Image.open(os.path.join(folder_path, img)) for img in batch_files]
        target_sizes = torch.tensor([img.size[::-1] for img in images]).to("cuda")

        inputs = processor(text=[texts] * len(images),
                         images=images, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        processed_outputs = processor.post_process_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes
        )

        # Filter boxes for each image in batch
        for idx, (filename, output) in enumerate(zip(batch_files, processed_outputs)):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            # Filter redundant boxes
            keep_indices = filter_redundant_boxes_owlv2(boxes, scores, labels)

            # Store filtered results
            filtered_output = {
                'boxes': output['boxes'][keep_indices],
                'scores': output['scores'][keep_indices],
                'labels': output['labels'][keep_indices]
            }

            results.append({
                'filename': filename,
                'results': filtered_output
            })

    return results

def visualize_detection(folder_path, results, image_idx, texts, processor):
    """Visualize detection results for a specific image"""
    image_path = os.path.join(folder_path, results[image_idx]['filename'])
    image = Image.open(image_path)

    boxes = results[image_idx]['results']['boxes']
    scores = results[image_idx]['results']['scores']
    labels = results[image_idx]['results']['labels']

    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image)

    # Draw boxes and labels
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = tuple(box)

        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red", width=1)
        text_label = f"{texts[label]} ({score:.2f})"
        draw.text(xy=(x1, y1-10), text=text_label, fill="white")

    return visualized_image

def save_yolo_labels(folder_path, results, texts):
    """
    Save detection results in YOLO format and return class mapping.
    Now includes filtering of redundant bounding boxes.
    """
    labels_dir = os.path.join(folder_path, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    class_mapping = {text: idx for idx, text in enumerate(texts)}

    for result in results:
        image_path = os.path.join(folder_path, result['filename'])
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        label_filename = os.path.splitext(result['filename'])[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        boxes = result['results']['boxes']
        labels = result['results']['labels']
        scores = result['results']['scores']

        # Convert boxes to YOLO format for filtering
        yolo_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Calculate normalized center coordinates and dimensions
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_boxes.append([x_center, y_center, width, height])

        # Convert to numpy arrays for filtering
        yolo_boxes = np.array(yolo_boxes)
        np_labels = np.array([label.item() for label in labels])
        np_scores = np.array([score.item() for score in scores])

        # Filter out redundant boxes
        if len(yolo_boxes) > 0:
            keep_indices = filter_redundant_boxes_owlv2(yolo_boxes, np_scores, np_labels)

            # Write filtered boxes to file
            with open(label_path, 'w') as f:
                for idx in keep_indices:
                    box = yolo_boxes[idx]
                    label = np_labels[idx]
                    f.write(f"{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        else:
            # Create empty file if no boxes
            open(label_path, 'w').close()

    return class_mapping

print("Beginning image labelling...")
results = process_image_folder(zip_file_images_folder,
                               processor, model, class_list)

print("Saving labels to disk in YOLO format...")
# Save labels and get class mapping
class_mapping = save_yolo_labels(zip_file_images_folder, results, class_list)
print("\nClass Mapping:")
for class_name, class_id in class_mapping.items():
    print(f"{class_name}: {class_id}")