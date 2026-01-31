import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stewart_fk_dataset_version_1/train.csv")  # change path if needed

# Extensions histograms
cols_e = ["e1","e2","e3","e4","e5","e6"]
df[cols_e].hist(bins=50, figsize=(12,6))
plt.suptitle("Actuator extension distributions (e1..e6)")
plt.tight_layout()
plt.show()

# Pose histograms
cols_pose = ["x","y","z","roll","pitch","yaw"]
df[cols_pose].hist(bins=50, figsize=(12,6))
plt.suptitle("Pose distributions (x,y,z,roll,pitch,yaw)")
plt.tight_layout()
plt.show()
