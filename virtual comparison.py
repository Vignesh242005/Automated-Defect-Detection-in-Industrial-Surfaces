import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_dir = "C:/Users/Vignesh/Downloads/processed_results"
os.makedirs(output_dir, exist_ok=True)  


df = pd.read_csv("C:/Users/Vignesh/Downloads/segmentation_metrics.csv")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["IoU"], bins=20, kde=True, color='blue')
plt.title("IoU Score Distribution")
plt.xlabel("IoU Score")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(df["Dice"], bins=20, kde=True, color='green')
plt.title("Dice Score Distribution")
plt.xlabel("Dice Score")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "iou_dice_distribution.png"))
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(df.index, df["IoU"], label="IoU", marker="o", linestyle="-", color="blue")
plt.plot(df.index, df["Dice"], label="Dice", marker="s", linestyle="--", color="green")
plt.xlabel("Image Index")
plt.ylabel("Score")
plt.title("IoU and Dice Scores Over Images")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "iou_dice_trend.png"))
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=df[["IoU", "Dice"]])
plt.title("IoU and Dice Score Boxplot")
plt.ylabel("Score")
plt.savefig(os.path.join(output_dir, "iou_dice_boxplot.png"))
plt.show()
