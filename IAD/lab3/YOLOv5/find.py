import cv2
import torch
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

input_dir = './test'
output_dir = './outputs'

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        print(f"Processing file: {filename}")

        # Read image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        # Perform inference
        print(f"Running inference on: {filename}")
        results = model(img)

        # Convert results to image with bounding boxes
        result_img = results.render()[0]

        # Convert BGR to RGB for OpenCV compatibility
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Save the output image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_img)
        print(f"Saved output to: {output_path}")

print("Detection complete. Check the output directory for results.")