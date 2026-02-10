import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/v1_clean/train.csv")

cols_e = ["e1", "e2", "e3", "e4", "e5", "e6"]
cols_titles = ["l1", "l2", "l3", "l4", "l5", "l6"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(cols_e):
    axes[i].hist(df[col], bins=50)
    axes[i].set_title(cols_titles[i])
    axes[i].set_xlabel("Extension (m)")
    axes[i].set_ylabel("Count")

fig.suptitle("Actuator extension distributions (l1–l6)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


cols_pose = ["x", "y", "z", "roll", "pitch", "yaw"]
units = ["mm", "mm", "mm", "rad", "rad", "rad"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
axes = axes.flatten()

for i, (col, unit) in enumerate(zip(cols_pose, units)):
    axes[i].hist(df[col], bins=50)
    axes[i].set_title(col)
    axes[i].set_xlabel(f"{col} ({unit})")
    axes[i].set_ylabel("Count")

fig.suptitle("Pose distributions (x, y, z, roll, pitch, yaw)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
