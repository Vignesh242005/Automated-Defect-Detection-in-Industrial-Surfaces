import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from skimage import filters, measure
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor


output_dir = "C:/Users/Vignesh/Downloads/KolektorSDD2"
os.makedirs(output_dir, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)


csv_file = os.path.join(output_dir, "segmentation_metrics.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "IoU", "Dice"])


def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ ERROR: Could not read image {img_path}")
    else:
        print(f"✅ Loaded image: {img_path}")
    return image

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def apply_filters(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    median = cv2.medianBlur(image, 5)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    return gaussian, median, bilateral

def kmeans_segmentation(image, clusters=3):
    img_flattened = image.reshape((-1, 1))
    scaler = MinMaxScaler()
    img_normalized = scaler.fit_transform(img_flattened)
    unique_values = np.unique(image)
    
    if len(unique_values) < clusters:
        print(f"⚠️ Warning: Skipping K-Means for image due to low variance.")
        return np.zeros_like(image)

    kmeans = KMeans(n_clusters=min(clusters, len(unique_values)), random_state=42, n_init=10)

    kmeans.fit(img_normalized)
    
    return kmeans.labels_.reshape(image.shape)

def region_growing(image):
    threshold = filters.threshold_otsu(image)
    binary = image > threshold
    labels = measure.label(binary, connectivity=2)
    return binary, labels


def edge_detection(image):
    return cv2.Canny(image, 100, 200)


def morphological_processing(binary_image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_image.astype(np.uint8), kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def evaluate_segmentation(ground_truth, predicted):
    gt_binary = ground_truth > 0
    pred_binary = predicted > 0
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    iou = intersection / union if union != 0 else 0
    dice = 2 * intersection / (gt_binary.sum() + pred_binary.sum() + 1e-6)
    return iou, dice



def visualize_comparison(original, segmented, ground_truth, filename, iou, dice):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title("Original")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(segmented, cmap='jet')
    ax2.set_title(f"Segmented\nIoU: {iou:.4f}, Dice: {dice:.4f}")
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(ground_truth, cmap='gray')
    ax3.set_title("Ground Truth")
    ax3.axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(save_path)
    print(f"✅ Saved visualization: {save_path}")

    plt.close(fig)  



def process_image(img_path, gt_path, dataset_type):
    filename = os.path.basename(img_path)
    image = load_image(img_path)
    ground_truth = load_image(gt_path) if os.path.exists(gt_path) else None
    
    if image is None:
        print(f"Skipping unreadable file: {filename}")
        return

    print(f"Processing: {filename}")

    
    hist_eq = histogram_equalization(image)
    gaussian, median, bilateral = apply_filters(hist_eq)

    
    segmented = kmeans_segmentation(hist_eq)
    binary, labels = region_growing(hist_eq)
    edges = edge_detection(hist_eq)
    morphed = morphological_processing(binary)

    
    for img, suffix in zip([hist_eq, edges, morphed], ["hist_eq", "edges", "morphed"]):
        save_path = os.path.join(output_dir, f"{dataset_type}_{filename}_{suffix}.png")
        if not cv2.imwrite(save_path, img):
            print(f"⚠️ Failed to save {save_path}")

    
    contours = find_contours(binary)
    contour_image = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{dataset_type}_{filename}_contours.png"), contour_image)

    
    if ground_truth is not None:
        iou, dice = evaluate_segmentation(ground_truth, segmented)
        print(f"{filename}: IoU={iou:.4f}, Dice={dice:.4f}")

        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, iou, dice])

        
        visualize_comparison(image, segmented, ground_truth, filename, iou, dice)
    else:
        print(f"⚠️ Ground truth not found for {filename}. Skipping evaluation.")

    print(f"✅ Processed {filename}")


def process_dataset(img_folder, gt_folder, dataset_type="train"):
    if not os.path.exists(img_folder):
        print(f"❌ Error: Folder {img_folder} not found!")
        return

    image_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = sorted([os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) != len(gt_files):
        print("⚠️ Warning: Number of images and ground truth files do not match!")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda args: process_image(*args, dataset_type), zip(image_files, gt_files))

train_folder = "C:/Users/Vignesh/Downloads/KolektorSDD2/train"
train_gt_folder = "C:/Users/Vignesh/Downloads/KolektorSDD2/train_gt"
test_folder = "C:/Users/Vignesh/Downloads/KolektorSDD2/test"
test_gt_folder = "C:/Users/Vignesh/Downloads/KolektorSDD2/test_gt"


print("Processing Training Data...")
process_dataset(train_folder, train_gt_folder, dataset_type="train")

print("Processing Testing Data...")
process_dataset(test_folder, test_gt_folder, dataset_type="test")

print(f"✅ Processed images and metrics saved in {output_dir}")
