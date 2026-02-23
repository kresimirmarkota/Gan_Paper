import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import cv2
import tensorflow as tf
import random
import pickle

from tensorflow import keras
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Dropout,
                                     Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization)

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
NOISE_DIM = 100
BATCH_SIZE = 16
BUFFER_SIZE = 60000

BASE_OUTPUT_DIR = "/content/dcgan_outputs"
IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
CHECKPOINTS_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_generator_model():
    """
    DCGAN Generator: Uses transposed convolutions with batch normalization
    Generates 200x200 RGB images from random noise
    """
    model = keras.Sequential(name='DCGAN_Generator')

    model.add(Dense(25 * 25 * 512, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((25, 25, 512)))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', activation='tanh'))

    return model


def make_discriminator_model():
    """
    DCGAN Discriminator: Uses strided convolutions instead of pooling
    Classifies whether images are real or fake
    """
    model = keras.Sequential(name='DCGAN_Discriminator')

    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(200, 200, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    """Binary cross-entropy loss for discriminator"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """Binary cross-entropy loss for generator"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """Single training step for DCGAN"""
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def update_lr(optimizer, factor, min_lr):
    """Reduce learning rate by factor"""
    old_lr = float(optimizer.learning_rate.numpy())
    new_lr = max(old_lr * factor, min_lr)
    optimizer.learning_rate.assign(new_lr)
    print(f"LR updated: {old_lr:.6f} -> {new_lr:.6f}")


def train(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer,
          epochs, checkpoint_manager, test_input, gen_loss_history, disc_loss_history,
          patience=5, factor=0.9, min_lr=1e-6, lr_start_epoch=100):
    """Train DCGAN model"""
    best_gen_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Starting epoch: {epoch + 1}/{epochs}")
        epoch_gen_loss = []
        epoch_disc_loss = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator,
                                             generator_optimizer, discriminator_optimizer)
            epoch_gen_loss.append(gen_loss.numpy())
            epoch_disc_loss.append(disc_loss.numpy())

        mean_gen_loss = np.mean(epoch_gen_loss)
        mean_disc_loss = np.mean(epoch_disc_loss)
        gen_loss_history.append(mean_gen_loss)
        disc_loss_history.append(mean_disc_loss)

        if epoch >= lr_start_epoch:
            if mean_gen_loss < best_gen_loss - 1e-4:
                best_gen_loss = mean_gen_loss
                epochs_no_improve = 0
                print(f"New best G_loss: {mean_gen_loss:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                update_lr(generator_optimizer, factor, min_lr)
                update_lr(discriminator_optimizer, factor, min_lr)
                epochs_no_improve = 0

        if (epoch + 1) % 10 == 0:
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, test_input)

        print(f'Epoch {epoch + 1}/{epochs} | G_loss={mean_gen_loss:.4f}, D_loss={mean_disc_loss:.4f}')


def generate_and_save_images(model, epoch, test_input):
    """Generate and save sample images"""
    predictions = model(test_input, training=False)

    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = predictions[i].numpy()
        plt.imshow(img)
        plt.axis("off")

    # plt.savefig(f"dcgan_image_at_epoch_{epoch:04d}.png")
    plt.savefig(os.path.join(IMAGES_DIR, f"dcgan_image_at_epoch_{epoch:04d}.png"))
    plt.close()
    print(f"Generated images saved for epoch {epoch}")


if __name__ == "__main__":

    X_loaded = pickle.load(open("/content/sample_data/X.pickle", "rb"))

    X = np.array(X_loaded).astype(np.float32)
    X = X.reshape(-1, 200, 200, 3)

    X = (X / 127.5) - 1

    print("DCGAN Training Setup")
    print("=" * 50)
    print(f"Data shape: {X.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Noise dimension: {NOISE_DIM}")
    print("=" * 50)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    print("\nGenerator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_input = tf.random.normal([16, NOISE_DIM])

    checkpoint_dir = CHECKPOINTS_DIR
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    gen_loss_history = []
    disc_loss_history = []
    EPOCHS = 200

    train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer,
          EPOCHS, ckpt_manager, test_input, gen_loss_history, disc_loss_history,
          patience=10, factor=0.9, min_lr=1e-6, lr_start_epoch=100)

    generate_and_save_images(generator, EPOCHS, test_input)

    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss_history, label='Generator loss', linewidth=2)
    plt.plot(disc_loss_history, label='Discriminator loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "dcgan_training_loss.png"))
    plt.show()

    print("\nDCGAN training completed!")



    generator = make_generator_model()


    discriminator = make_discriminator_model()


    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint,
        CHECKPOINTS_DIR,
        max_to_keep=3
    )

    import shutil
    # from google.colab import files

    zip_path = "/content/dcgan_outputs.zip"

    shutil.make_archive(
        base_name="/content/dcgan_outputs",
        format="zip",
        root_dir=BASE_OUTPUT_DIR
    )

    files.download(zip_path)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("✅ Checkpoint restored!")
    else:
        raise ValueError("❌ Nema checkpointa za učitati!")

    NUM_IMAGES = 10
    TEST_DIR = "/content/dcgan_outputs"
    noise = tf.random.normal([NUM_IMAGES, NOISE_DIM])

    generated_images = generator(noise, training=False)

    # [-1, 1] -> [0, 1]
    generated_images = (generated_images + 1) / 2.0
    plt.figure(figsize=(15, 3))

    for i in range(NUM_IMAGES):
        img = generated_images[i].numpy()

        # save
        save_path = os.path.join(TEST_DIR, f"generated_test_{i + 1}.png")
        plt.imsave(save_path, img)

        # show
        plt.subplot(1, NUM_IMAGES, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.suptitle("Generated DCGAN Test Images", fontsize=14)
    plt.show()

    print(f"🖼️ {NUM_IMAGES} novih slika spremljeno u:\n{TEST_DIR}")
    print("Min pixel:", generated_images.numpy().min())
    print("Max pixel:", generated_images.numpy().max())