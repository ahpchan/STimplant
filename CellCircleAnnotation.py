from tkinter import Tk, filedialog, Label, Button, Scale, HORIZONTAL, Entry, StringVar, OptionMenu, Text, Scrollbar, END, Toplevel, Canvas
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import os
import csv
import random
import pandas as pd
from sklearn.cluster import DBSCAN


# Initialize color list
colors = []
rgb_colors = []
df = None
original_image = None
image_window = None
canvas = None
photo = None


def get_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir
    count = 1
    while True:
        new_dir = f"{base_dir}({count})"
        if not os.path.exists(new_dir):
            return new_dir
        count += 1


def process_image(image_path, annotation_ratio, circle_radius):
    global original_image, df
    original_image = Image.open(image_path).convert("RGB")
    image_data = np.array(original_image)
    output_dir = get_unique_dir("annotated_images")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "annotated_points.csv")
    all_circle_info_path = os.path.join(output_dir, "all_circle_info.csv")

    with open(csv_path, mode="w", newline="", encoding="utf - 8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Color Index", "Color Hex", "X Coordinate", "Y Coordinate"])

        output_image_pil = original_image.copy()
        draw = ImageDraw.Draw(output_image_pil)
        font = ImageFont.load_default()

        all_annotated_centers = []
        for idx, color in enumerate(rgb_colors):
            mask = np.all(image_data == color, axis=-1)
            color_points = np.column_stack(np.where(mask))

            if len(color_points) > 0:
                clustering = DBSCAN(eps=5, min_samples=1).fit(color_points)
                cluster_centers = []
                for cluster_id in np.unique(clustering.labels_):
                    cluster = color_points[clustering.labels_ == cluster_id]
                    center = cluster.mean(axis=0) if len(cluster) > 0 else np.array([0, 0])
                    cluster_centers.append(center)
                cluster_centers = np.array(cluster_centers).astype(int)
            else:
                cluster_centers = np.empty((0, 2))

            num_points_to_annotate = int(len(cluster_centers) * annotation_ratio)
            if num_points_to_annotate > 0:
                annotated_indices = random.sample(range(len(cluster_centers)), num_points_to_annotate)
            else:
                annotated_indices = []

            for i, center in enumerate(cluster_centers):
                y, x = center
                csv_writer.writerow([idx + 1, colors[idx], x, y])
                if i in annotated_indices:
                    draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                                 outline="red", width=2)
                    draw.text((x + circle_radius + 2, y), f"C{idx + 1}", fill="white", font=font)
                    all_annotated_centers.append((idx + 1, colors[idx], x, y))

        output_image_pil.save(os.path.join(output_dir, "annotated_image.png"))

    df = pd.read_csv(csv_path)

    with open(all_circle_info_path, mode="w", newline="", encoding="utf - 8") as all_circle_csv:
        fieldnames = ['Circle Color Index', 'Circle Color Hex', 'Circle Center X', 'Circle Center Y', 'Inner Color Index',
                      'Inner Color Hex', 'Count']
        writer = csv.DictWriter(all_circle_csv, fieldnames=fieldnames)
        writer.writeheader()

        for color_index, color_hex, center_x, center_y in all_annotated_centers:
            circle_points = df[(df['X Coordinate'] >= center_x - circle_radius) & (df['X Coordinate'] <= center_x + circle_radius) &
                               (df['Y Coordinate'] >= center_y - circle_radius) & (df['Y Coordinate'] <= center_y + circle_radius)]
            color_counts = circle_points.groupby(['Color Index', 'Color Hex']).size().reset_index(name='Count')

            for _, row in color_counts.iterrows():
                writer.writerow({
                    'Circle Color Index': color_index,
                    'Circle Color Hex': color_hex,
                    'Circle Center X': center_x,
                    'Circle Center Y': center_y,
                    'Inner Color Index': row['Color Index'],
                    'Inner Color Hex': row['Color Hex'],
                    'Count': row['Count']
                })

    return df


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All Files", "*.*")])
    if file_path:
        label_file.config(text=f"Selected file: {file_path}")
        btn_process.config(state="normal", command=lambda: process_with_ratio(file_path))


def load_color_reference():
    global colors, rgb_colors
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if file_path:
        color_data = pd.read_excel(file_path)
        if len(color_data.columns) >= 2:
            colors = color_data.iloc[:, 1].tolist()
            rgb_colors = [tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) for hex_color in colors]
            label_color_ref.config(text=f"Loaded {len(colors)} colors from reference file.")
        else:
            label_color_ref.config(text="Invalid color reference file format.")


def process_with_ratio(file_path):
    global df
    annotation_ratio = scale.get() / 100
    try:
        circle_radius = int(entry_radius.get())
    except ValueError:
        label_result.config(text="Invalid radius value. Please enter a number.")
        return
    df = process_image(file_path, annotation_ratio, circle_radius)
    update_dropdown(df)
    label_result.config(text="Processing complete! Check the output folder.")
    show_image()


def update_dropdown(df):
    global selected_point_var
    points = [f"Color {row['Color Index']}, Point at ({row['X Coordinate']}, {row['Y Coordinate']})" for _, row in df.iterrows()]
    selected_point_var.set(points[0])
    dropdown = OptionMenu(root, selected_point_var, *points, command=show_circle_info)
    dropdown.pack(pady=10)


def show_circle_info(selected_point):
    global df, original_image, canvas, photo
    point_strings = [f"Color {row['Color Index']}, Point at ({row['X Coordinate']}, {row['Y Coordinate']})" for _, row in df.iterrows()]
    index = point_strings.index(selected_point)
    center_x = df.iloc[index]['X Coordinate']
    center_y = df.iloc[index]['Y Coordinate']
    circle_radius = int(entry_radius.get())
    circle_points = df[(df['X Coordinate'] >= center_x - circle_radius) & (df['X Coordinate'] <= center_x + circle_radius) &
                       (df['Y Coordinate'] >= center_y - circle_radius) & (df['Y Coordinate'] <= center_y + circle_radius)]
    color_counts = circle_points.groupby(['Color Index', 'Color Hex']).size().reset_index(name='Count')

    text_output.delete("1.0", END)
    text_output.insert(END, "Color Frequency in Selected Area:\n")
    text_output.insert(END, color_counts.to_string(index=False))

    # 在原图上绘制高亮圆圈
    temp_image = original_image.copy()
    draw = ImageDraw.Draw(temp_image)
    draw.ellipse([(center_x, center_y), (center_x + 2 * circle_radius, center_y + 2 * circle_radius)],
                 outline="yellow", width=4)  # 用黄色粗线突出显示
    show_image(temp_image)


def show_image(img=None):
    global image_window, canvas, photo

    if not image_window:
        if img is None and original_image:
            img = original_image

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        win_width = screen_width // 3
        win_height = screen_height // 3

        image_window = Toplevel(root)
        image_window.geometry(f"{win_width}x{win_height}")
        image_window.title("Image Display")

        frame = Canvas(image_window)
        frame.pack(fill='both', expand=True)

        vbar = Scrollbar(frame, orient='vertical')
        vbar.pack(side='right', fill='y')
        hbar = Scrollbar(frame, orient='horizontal')
        hbar.pack(side='bottom', fill='x')

        canvas = Canvas(frame, yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        vbar.config(command=canvas.yview)
        hbar.config(command=canvas.xview)

        image_window.bind("<Configure>", lambda event: update_resized_image(img))

    if img is None and original_image:
        img = original_image
    update_resized_image(img)


def update_resized_image(img):
    """ Resize image dynamically based on window size """
    global image_window, canvas, photo

    if image_window:
        win_width = image_window.winfo_width()
        win_height = image_window.winfo_height()

        img_aspect = img.width / img.height
        new_width = win_width - 20 
        new_height = int(new_width / img_aspect)

        if new_height > win_height - 20:
            new_height = win_height - 20
            new_width = int(new_height * img_aspect)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_img)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.config(scrollregion=canvas.bbox("all"))


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
scale.set(10)
scale.pack()

Label(root, text="Step 4: Enter circle radius").pack(pady=10)
entry_radius = Entry(root)
entry_radius.insert(0, "5")
entry_radius.pack()

btn_process = Button(root, text="Start Processing", state="disabled")
btn_process.pack(pady=20)
label_result = Label(root, text="")
label_result.pack(pady=10)
selected_point_var = StringVar(root)

text_output = Text(root, height=10, width=60)
text_output.pack(pady=10)

root.mainloop()