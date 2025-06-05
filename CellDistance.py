from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
from skimage.draw import polygon
from sklearn.cluster import DBSCAN
import random
import pandas as pd

colors = []
rgb_colors = []
additional_color = (0, 85, 0)  # This color could be set manually

def get_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir
    count = 1
    while True:
        new_dir = f"{base_dir}({count})"
        if not os.path.exists(new_dir):
            return new_dir
        count += 1

def process_image(image_path, annotation_ratio):
    image = Image.open(image_path).convert("RGB")
    image_data = np.array(image)
    output_dir = get_unique_dir("separated_colors")
    os.makedirs(output_dir, exist_ok=True)
    boundary_mask = np.all(image_data == additional_color, axis=-1)
    boundary_points = np.column_stack(np.where(boundary_mask))

    if boundary_points.size > 0:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(boundary_points)
        optimized_boundary = boundary_points[hull.vertices]

        rr, cc = polygon(optimized_boundary[:, 0], optimized_boundary[:, 1], image_data.shape[:2])
        optimized_boundary_mask = np.zeros_like(boundary_mask, dtype=bool)
        optimized_boundary_mask[rr, cc] = True
    else:
        optimized_boundary_mask = boundary_mask

    csv_path = os.path.join(output_dir, "color_points_data.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Color Index", "Color Hex", "X Coordinate", "Y Coordinate", "Distance"])

        # Iterate through the list of colors and create masks
        for idx, color in enumerate(rgb_colors):
            mask = np.all(image_data == color, axis=-1)
            color_points = np.column_stack(np.where(mask))

            if len(color_points) > 0:
                clustering = DBSCAN(eps=3, min_samples=1).fit(color_points)
                cluster_centers = []
                for cluster_id in np.unique(clustering.labels_):
                    cluster = color_points[clustering.labels_ == cluster_id]
                    center = cluster.mean(axis=0)
                    cluster_centers.append(center)
                cluster_centers = np.array(cluster_centers).astype(int)
            else:
                cluster_centers = np.empty((0, 2))

            if boundary_points.size > 0 and cluster_centers.size > 0:
                distances = cdist(cluster_centers, boundary_points).min(axis=1)
                inside_boundary = optimized_boundary_mask[cluster_centers[:, 0], cluster_centers[:, 1]]
                distances[~inside_boundary] *= -1
            else:
                distances = np.array([])

            if len(distances) > 0:
                num_points_to_annotate = int(len(cluster_centers) * annotation_ratio)
                if num_points_to_annotate > 0:
                    annotated_indices = random.sample(range(len(cluster_centers)), num_points_to_annotate)
                else:
                    annotated_indices = []

            output_image = np.zeros_like(image_data, dtype=np.uint8)

            output_image[mask] = image_data[mask]

            output_image[optimized_boundary_mask] = additional_color

            output_image_pil = Image.fromarray(output_image)
            draw = ImageDraw.Draw(output_image_pil)
            font = ImageFont.load_default()

            num_points_to_annotate = int(len(cluster_centers) * annotation_ratio)
            if num_points_to_annotate > 0:
                annotated_indices = random.sample(range(len(cluster_centers)), num_points_to_annotate)
            else:
                annotated_indices = []

            for i, (center, distance) in enumerate(zip(cluster_centers, distances)):
                y, x = center
                csv_writer.writerow([idx + 1, colors[idx], x, y, distance])
                if i in annotated_indices:
                    marker_radius = 5
                    draw.ellipse(
                        [(x - marker_radius, y - marker_radius), (x + marker_radius, y + marker_radius)],
                        outline="red",
                        width=2
                    )
                    draw.text((x + marker_radius + 2, y), f"{distance:.1f}", fill="white", font=font)

            output_image_pil.save(os.path.join(output_dir, f"color_{idx + 1}.png"))

    print(f"Separated color images with annotations have been saved to: {output_dir}")
    print(f"CSV file with point data has been saved to: {csv_path}")

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), 
    ("All Files", "*.*")])
    if file_path:
        label_file.config(text=f"Selected file: {file_path}")
        btn_process.config(state="normal")
        btn_process.config(command=lambda: process_with_ratio(file_path))

def load_color_reference():
    global colors, rgb_colors
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if file_path:
        color_data = pd.read_excel(file_path)
        if len(color_data.columns) >= 2:
            colors = color_data.iloc[:, 1].tolist()
            rgb_colors = [tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5)) for hex_color in colors]
            label_color_ref.config(text=f"Loaded {len(colors)} colors from reference file.")
        else:
            label_color_ref.config(text="Invalid color reference file format.")

def process_with_ratio(file_path):
    annotation_ratio = scale.get() / 100  # Convert percentage to ratio
    process_image(file_path, annotation_ratio)
    label_result.config(text="Processing complete! Check the output folder.")

root = Tk()
root.title("Image Annotation Tool")

Label(root, text="Step 1: Load color reference file").pack(pady=10)
btn_load_colors = Button(root, text="Load Color Reference", command=load_color_reference)
btn_load_colors.pack()

label_color_ref = Label(root, text="No color reference file loaded")
label_color_ref.pack(pady=5)

Label(root, text="Step 2: Select an image").pack(pady=10)
btn_open = Button(root, text="Open Image", command=open_file)
btn_open.pack()

label_file = Label(root, text="No file selected")
label_file.pack(pady=5)

Label(root, text="Step 3: Set annotation percentage").pack(pady=10)
scale = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="Annotation %")
scale.set(10)  # Default value
scale.pack()

btn_process = Button(root, text="Start Processing", state="disabled")
btn_process.pack(pady=20)

label_result = Label(root, text="")
label_result.pack(pady=10)

root.mainloop()
