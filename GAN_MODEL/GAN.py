import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout

# Configuration
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
NOISE_DIM = 100
BATCH_SIZE = 8
BUFFER_SIZE = 60000

INITIAL_LR = 2e-4
MIN_LR = 1e-6
LR_DECAY_FACTOR = 0.5
PATIENCE = 10
LR_START_EPOCH = 20

BASE_OUTPUT_DIR = "/content/vanillagan3_outputs3"
IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
CHECKPOINTS_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_generator_model():
    model = keras.Sequential(name='VanillaGAN_Generator')

    model.add(Dense(256, input_shape=(NOISE_DIM,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(IMAGE_WIDTH * IMAGE_HEIGHT * 3, activation='tanh'))

    model.add(Reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3)))

    return model


def make_discriminator_model():
    model = keras.Sequential(name='VanillaGAN_Discriminator')

    model.add(Flatten(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def update_learning_rate(optimizer, new_lr):
    old_lr = float(optimizer.learning_rate.numpy())
    optimizer.learning_rate.assign(new_lr)
    return old_lr


@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

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

    return gen_loss, disc_loss, real_output, fake_output


def train(dataset, generator, discriminator, generator_optimizer, discriminator_optimizer,
          epochs, checkpoint_manager, test_input, gen_loss_history, disc_loss_history):
    best_gen_loss = float('inf')
    epochs_without_improvement = 0
    lr_history = []

    for epoch in range(epochs):
        print(f"Starting epoch: {epoch + 1}/{epochs}")
        epoch_gen_loss = []
        epoch_disc_loss = []

        for image_batch in dataset:
            gen_loss, disc_loss, _, _ = train_step(
                image_batch, generator, discriminator,
                generator_optimizer, discriminator_optimizer
            )
            epoch_gen_loss.append(gen_loss.numpy())
            epoch_disc_loss.append(disc_loss.numpy())

        mean_gen_loss = np.mean(epoch_gen_loss)
        mean_disc_loss = np.mean(epoch_disc_loss)
        gen_loss_history.append(mean_gen_loss)
        disc_loss_history.append(mean_disc_loss)

        current_lr = float(generator_optimizer.learning_rate.numpy())
        lr_history.append(current_lr)

        if epoch >= LR_START_EPOCH:

            if mean_gen_loss < best_gen_loss - 0.01:
                best_gen_loss = mean_gen_loss
                epochs_without_improvement = 0
                print(f"  ✓ New best generator loss: {best_gen_loss:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"  ⏸ No improvement for {epochs_without_improvement} epochs")

            if epochs_without_improvement >= PATIENCE:
                new_gen_lr = max(current_lr * LR_DECAY_FACTOR, MIN_LR)
                new_disc_lr = max(current_lr * LR_DECAY_FACTOR, MIN_LR)

                old_gen_lr = update_learning_rate(generator_optimizer, new_gen_lr)
                old_disc_lr = update_learning_rate(discriminator_optimizer, new_disc_lr)

                print(f"  🔽 LEARNING RATE REDUCED!")
                print(f"     Generator LR: {old_gen_lr:.6f} → {new_gen_lr:.6f}")
                print(f"     Discriminator LR: {old_disc_lr:.6f} → {new_disc_lr:.6f}")

                # Reset counter
                epochs_without_improvement = 0

                best_gen_loss = mean_gen_loss
        else:
            print(f"  ⏳ Waiting for epoch {LR_START_EPOCH} to start LR scheduling")

        if (epoch + 1) % 10 == 0:
            checkpoint_manager.save()
            print(f"  💾 Checkpoint saved at epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, test_input)

        print(
            f'Epoch {epoch + 1}/{epochs} | G_loss={mean_gen_loss:.4f}, D_loss={mean_disc_loss:.4f}, LR={current_lr:.6f}')
        print("-" * 80)

    return lr_history


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = predictions[i].numpy()
        plt.imshow(img)
        plt.axis("off")

    plt.savefig(os.path.join(IMAGES_DIR, f"vanillagan_image_at_epoch_{epoch:04d}.png"))
    plt.close()
    print(f"  🖼️  Generated images saved for epoch {epoch}")


if __name__ == "__main__":

    X_loaded = pickle.load(open("/content/sample_data/X.pickle", "rb"))

    X = np.array(X_loaded).astype(np.float32)
    X = X.reshape(-1, 200, 200, 3)

    X = (X / 127.5) - 1

    print("Vanilla GAN Training Setup (with Adaptive LR)")
    print("=" * 50)
    print(f"Data shape: {X.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Noise dimension: {NOISE_DIM}")
    print("=" * 50)
    print("\nLearning Rate Scheduling:")
    print(f"  Initial LR: {INITIAL_LR}")
    print(f"  Minimum LR: {MIN_LR}")
    print(f"  Decay factor: {LR_DECAY_FACTOR}")
    print(f"  Patience: {PATIENCE} epochs")
    print(f"  Start monitoring: Epoch {LR_START_EPOCH}")
    print("=" * 50)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    print("\nGenerator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, beta_1=0.5)

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

    # Restore checkpoint if exists
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    # Training
    gen_loss_history = []
    disc_loss_history = []
    EPOCHS = 200

    lr_history = train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer,
                       EPOCHS, ckpt_manager, test_input, gen_loss_history, disc_loss_history)

    # Final image generation
    generate_and_save_images(generator, EPOCHS, test_input)

    # Plot loss curves with learning rate
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Losses
    axes[0].plot(gen_loss_history, label='Generator loss', linewidth=2, color='blue')
    axes[0].plot(disc_loss_history, label='Discriminator loss', linewidth=2, color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Vanilla GAN Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Learning Rate
    axes[1].plot(lr_history, label='Learning Rate', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "vanillagan_training_with_lr.png"), dpi=300)
    plt.show()

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Final Generator Loss: {gen_loss_history[-1]:.4f}")
    print(f"Final Discriminator Loss: {disc_loss_history[-1]:.4f}")
    print(f"Final Learning Rate: {lr_history[-1]:.6f}")
    print(f"Initial Learning Rate: {lr_history[0]:.6f}")
    print(f"LR Reduction: {lr_history[0] / lr_history[-1]:.2f}x")
    print("=" * 50)

    print("\nVanilla GAN training completed!")

    import shutil
    # from google.colab import files

    ZIP_PATH = "/content/vanillagan2_outputs2.zip"

    shutil.make_archive(
        base_name="/content/vanillagan2_outputs2",
        format="zip",
        root_dir=BASE_OUTPUT_DIR
    )

    files.download(ZIP_PATH)

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

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("✅ Checkpoint restored!")
    else:
        raise ValueError("❌ Nema checkpointa za učitati!")

    NUM_IMAGES = 10
    TEST_DIR = "/content/vanillagan2_outputs2"
    noise = tf.random.normal([NUM_IMAGES, NOISE_DIM])

    generated_images = generator(noise, training=False)

    generated_images = (generated_images + 1) / 2.0
    plt.figure(figsize=(15, 3))

    for i in range(NUM_IMAGES):
        img = generated_images[i].numpy()

        save_path = os.path.join(TEST_DIR, f"generated_test_{i + 1}.png")
        plt.imsave(save_path, img)

        plt.subplot(1, NUM_IMAGES, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.suptitle("Generated vanillagan Test Images", fontsize=14)
    plt.show()

    print(f"🖼️ {NUM_IMAGES} novih slika spremljeno u:\n{TEST_DIR}")
    print("Min pixel:", generated_images.numpy().min())
    print("Max pixel:", generated_images.numpy().max())