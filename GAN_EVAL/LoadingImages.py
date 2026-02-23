# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import os
# import cv2
# import tensorflow as tf
# import random
# import numpy as np
# import pickle
#
# folder_path = r'C:\Users\kmark\Desktop\DeepLearning\archive\Grape'
# DATADIR = r'C:\Users\kmark\Desktop\DeepLearning\archive\Grape'
# CATEGORIES = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]
#
# IMG_SIZE = 200
# training_data = []
#
# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_num = CATEGORIES.index(category)
#         img_list = os.listdir(path)
#         # Uzmi svaki drugi fajl (korak 2) za upola manje slika
#         for i in range(0, len(img_list), 1):
#             img = img_list[i]
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGRA2BGR)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 training_data.append([new_array, class_num])
#             except Exception as e:
#                 pass
#
# create_training_data()
#
# random.shuffle(training_data)
#
# X = []
# y = []
#
# for features, label in training_data:
#     X.append(features)
#     y.append(label)
#
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# y = np.array(y)
#
# # Spremi u pickle
# with open("X.pickle", "wb") as pickle_out:
#     pickle.dump(X, pickle_out)
#
# with open("y.pickle", "wb") as pickle_out:
#     pickle.dump(y, pickle_out)
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Folder u koji spremamo slike
output_folder = r"C:\Users\kmark\Desktop\DeepLearning\Saved_Images"
os.makedirs(output_folder, exist_ok=True)

# --- 1. Učitaj originalne slike iz dataseta ---
dataset_folder = r'C:\Users\kmark\Desktop\DeepLearning\archive\Grape'
CATEGORIES = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

original_images = []
labels_orig = []

for idx, category in enumerate(CATEGORIES):
    class_folder = os.path.join(dataset_folder, category)
    file_names = os.listdir(class_folder)
    for f in file_names[:1]:  # uzmi po jednu sliku po klasi za primjer
        img_path = os.path.join(class_folder, f)
        img = cv2.imread(img_path)  # BGR format
        original_images.append(img)
        labels_orig.append(idx)

original_images = np.array(original_images)
labels_orig = np.array(labels_orig)

# Prikaz originalnih slika
fig_orig, axes_orig = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    img_rgb = original_images[i][:, :, ::-1]  # BGR -> RGB
    axes_orig[i].imshow(img_rgb)
    axes_orig[i].set_title(f"{CATEGORIES[labels_orig[i]]}", fontsize=10)
    axes_orig[i].axis('off')

plt.tight_layout()
plt.suptitle("Originalne slike iz dataseta", fontsize=16, y=1.05)

# Spremi originalne slike kao jednu figuru
output_path_orig = os.path.join(output_folder, "original_images.png")
plt.savefig(output_path_orig, dpi=300, bbox_inches='tight')
plt.show()


# --- 2. Učitaj predobrađene slike iz pickle-a i normaliziraj ---
with open("X.pickle", "rb") as f:
    X = pickle.load(f)
with open("y.pickle", "rb") as f:
    y = pickle.load(f)

# Normalizacija [0,1] i konverzija u float32
X_normalized = X.astype(np.float32) / 255.0

# Prikaz 4 predobrađenih slika
fig_proc, axes_proc = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    img_rgb = X_normalized[i][:, :, ::-1]  # BGR -> RGB
    axes_proc[i].imshow(img_rgb)
    axes_proc[i].set_title(f"{CATEGORIES[y[i]]}", fontsize=10)
    axes_proc[i].axis('off')

plt.tight_layout()
plt.suptitle("Predobrađene slike iz pickle-a (200x200 + normalizirane)", fontsize=16, y=1.05)

# Spremi predobrađene slike kao jednu figuru
output_path_proc = os.path.join(output_folder, "preprocessed_images.png")
plt.savefig(output_path_proc, dpi=300, bbox_inches='tight')
plt.show()

print(f"Slike spremljene u folder: {output_folder}")