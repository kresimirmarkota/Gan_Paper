
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.applications.inception_v3 import InceptionV3, preprocess_input

from scipy.linalg import sqrtm
from scipy.stats import entropy
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Conv2D, Conv2DTranspose,
                     LeakyReLU, BatchNormalization, Dropout)
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from tqdm import tqdm


from scipy.linalg import sqrtm
# --- CONFIG ---
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
NOISE_DIM = 100
NUM_IMAGES = 5000  # ← POVEĆAJ!
BATCH_SIZE = 8

inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
# ... (ostalo ostaje isto)

# --- Generate Images (bez konverzije) ---
def generate_images(generator, num_images=NUM_IMAGES):
    noise = tf.random.normal([num_images, NOISE_DIM])
    images = generator(noise, training=False)
    # Ostavi u [-1,1] - NE konvertiraj!
    return images

def calculate_fid(real_images, fake_images):
    act_real = inception_model.predict(preprocess_for_inception(real_images), verbose=0)
    act_fake = inception_model.predict(preprocess_for_inception(fake_images), verbose=0)

    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)

    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def calculate_kid(real_images, fake_images):
    def polynomial_mmd(X, Y, degree=3, gamma=None, coef0=1):
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        K_XX = (gamma * X @ X.T + coef0) ** degree
        K_YY = (gamma * Y @ Y.T + coef0) ** degree
        K_XY = (gamma * X @ Y.T + coef0) ** degree
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    act_real = inception_model.predict(preprocess_for_inception(real_images), verbose=0)
    act_fake = inception_model.predict(preprocess_for_inception(fake_images), verbose=0)
    return polynomial_mmd(act_real, act_fake)
# --- IS funkcija s provjerom ---
def calculate_inception_score(images, splits=10):
    # Provjeri input
    print(f"IS input range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")

    # Konvertiraj u [0, 255]
    if images.max() <= 1.0 and images.min() >= 0.0:  # [0,1]
        images = images * 255.0
    elif images.min() < 0:  # [-1,1]
        images = (images + 1) * 127.5

    # Resize i preprocessing
    images = tf.image.resize(images, (299, 299))
    images = preprocess_input(images)

    # Inception predictions
    inception = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True, weights='imagenet'
    )
    preds = inception.predict(images, verbose=0, batch_size=32)

    # Debug info
    print(f"Predictions range: [{preds.min():.6f}, {preds.max():.6f}]")
    print(f"First pred sum: {preds[0].sum():.6f} (should be ~1.0)")

    scores = []
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl_divs = [entropy(p, py) for p in part]
        scores.append(np.exp(np.mean(kl_divs)))

    return np.mean(scores), np.std(scores)


# Generate
vanilla_images = generate_images(vanilla_gen)
dcgan_images = generate_images(dcgan_gen)
wgan_images = generate_images(wgan_gen)

# Calculate Metrics
real_images_sample = X[np.random.choice(len(X), NUM_IMAGES, replace=False)]
for name, fake_images in zip(["VanillaGAN", "DCGAN", "WGAN"],
                             [vanilla_images, dcgan_images, wgan_images]):
    print(f"\n=== {name} ===")
    fid = calculate_fid(real_images_sample, fake_images)
    kid = calculate_kid(real_images_sample, fake_images)
    is_mean, is_std = calculate_inception_score(fake_images)
    print(f"{name}: FID={fid:.2f}, KID={kid:.6f}, IS={is_mean:.2f} ± {is_std:.2f}")

# Display (konvertiraj u [0,1] SAMO za prikaz)
fig, axes = plt.subplots(3, 16, figsize=(32, 6))  # Prikaži prvih 16
model_names = ["VanillaGAN", "DCGAN", "WGAN"]
all_images = [vanilla_images[:16], dcgan_images[:16], wgan_images[:16]]

for i in range(3):
    for j in range(16):
        img = (all_images[i][j] + 1.0) / 2.0  # [-1,1] -> [0,1] za prikaz
        img = np.clip(img, 0, 1)
        axes[i, j].imshow(img)
        axes[i, j].axis('off')
    axes[i, 0].set_ylabel(model_names[i], fontsize=12)

plt.tight_layout()
plt.show()