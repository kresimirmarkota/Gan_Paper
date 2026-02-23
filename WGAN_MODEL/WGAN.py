import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
NOISE_DIM = 100
BATCH_SIZE = 16
BUFFER_SIZE = 60000
LAMBDA_GP = 10
N_CRITIC = 5

BASE_OUTPUT_DIR = "/content/wgan2_outputs2"
IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
CHECKPOINTS_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def make_generator_model():
    """
    WGAN Generator - same as before but with better initialization
    """
    model = keras.Sequential(name='WGAN_Generator')

    model.add(Dense(25 * 25 * 512, use_bias=False, input_shape=(NOISE_DIM,),
                    kernel_initializer='he_normal'))  # Better initialization
    model.add(Reshape((25, 25, 512)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                              use_bias=False, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                              use_bias=False, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',
                              use_bias=False, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', activation='tanh',
                              kernel_initializer='glorot_normal'))

    return model


def make_critic_model():
    model = keras.Sequential(name='WGAN_Critic')

    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                     input_shape=(200, 200, 3), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='glorot_normal'))  # No sigmoid!

    return model


def gradient_penalty(critic, real_images, fake_images, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_output = critic(interpolated, training=True)

    gradients = tape.gradient(critic_output, interpolated)

    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum + 1e-8)

    gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1.0))

    return gradient_penalty


def critic_loss(real_output, fake_output, gp):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = real_loss + fake_loss + LAMBDA_GP * gp
    return total_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


@tf.function
def train_critic_step(images, generator, critic, critic_optimizer):
    """
    Single training step for critic only
    """
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as critic_tape:
        generated_images = generator(noise, training=True)

        real_output = critic(images, training=True)
        fake_output = critic(generated_images, training=True)

        gp = gradient_penalty(critic, images, generated_images, batch_size)

        c_loss = critic_loss(real_output, fake_output, gp)

    gradients_of_critic = critic_tape.gradient(c_loss, critic.trainable_variables)

    gradients_of_critic = [tf.clip_by_norm(g, 1.0) for g in gradients_of_critic]
    critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    return c_loss, real_output, fake_output


@tf.function
def train_generator_step(batch_size, generator, critic, generator_optimizer):
    """
    Single training step for generator only
    """
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = critic(generated_images, training=True)
        g_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)

    gradients_of_generator = [tf.clip_by_norm(g, 1.0) for g in gradients_of_generator]
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return g_loss


def train(dataset, generator, critic, generator_optimizer, critic_optimizer,
          epochs, checkpoint_manager, test_input, gen_loss_history, critic_loss_history,
          n_critic=5):
    for epoch in range(epochs):
        print(f"Starting epoch: {epoch + 1}/{epochs}")
        epoch_gen_loss = []
        epoch_critic_loss = []

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]

            for _ in range(n_critic):
                c_loss, real_out, fake_out = train_critic_step(
                    image_batch, generator, critic, critic_optimizer
                )
                epoch_critic_loss.append(c_loss.numpy())

            g_loss = train_generator_step(batch_size, generator, critic, generator_optimizer)
            epoch_gen_loss.append(g_loss.numpy())

        mean_gen_loss = np.mean(epoch_gen_loss)
        mean_critic_loss = np.mean(epoch_critic_loss)
        gen_loss_history.append(mean_gen_loss)
        critic_loss_history.append(mean_critic_loss)

        if (epoch + 1) % 10 == 0:
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, test_input)

        print(f'Epoch {epoch + 1}/{epochs} | G_loss={mean_gen_loss:.4f}, C_loss={mean_critic_loss:.4f}')


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

    plt.savefig(os.path.join(IMAGES_DIR, f"wgan_image_at_epoch_{epoch:04d}.png"))
    plt.close()
    print(f"Generated images saved for epoch {epoch}")


if __name__ == "__main__":

    X_loaded = pickle.load(open("/content/sample_data/X.pickle", "rb"))

    X = np.array(X_loaded).astype(np.float32)
    X = X.reshape(-1, 200, 200, 3)

    X = ((X / 127.5) - 1)

    print("IMPROVED WGAN-GP Training Setup")
    print("=" * 50)
    print(f"Data shape: {X.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Noise dimension: {NOISE_DIM}")
    print(f"Gradient penalty coefficient: {LAMBDA_GP}")
    print(f"N_critic (critic updates per generator): {N_CRITIC}")
    print("=" * 50)

    generator = make_generator_model()
    critic = make_critic_model()

    print("\nGenerator Summary:")
    generator.summary()
    print("\nCritic Summary:")
    critic.summary()

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)

    # Or use Adam with conservative settings:
    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
    # critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

    train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_input = tf.random.normal([16, NOISE_DIM])

    checkpoint_dir = CHECKPOINTS_DIR
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        critic=critic,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    gen_loss_history = []
    critic_loss_history = []
    EPOCHS = 200

    print("\n" + "=" * 50)
    print("KEY IMPROVEMENTS:")
    print("=" * 50)
    print("1. Increased batch size: 8 → 16")
    print("2. Better weight initialization (He normal)")
    print("3. Gradient clipping for stability")
    print("4. RMSprop optimizer (alternative to Adam)")
    print("5. Proper gradient penalty implementation")
    print("6. Separated critic and generator training steps")
    print("=" * 50 + "\n")

    train(train_dataset, generator, critic, generator_optimizer, critic_optimizer,
          EPOCHS, ckpt_manager, test_input, gen_loss_history, critic_loss_history,
          n_critic=N_CRITIC)

    generate_and_save_images(generator, EPOCHS, test_input)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(gen_loss_history, label='Generator loss', linewidth=2, color='blue')
    plt.plot(critic_loss_history, label='Critic loss', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN-GP Training Loss (Improved)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    window = 10
    if len(gen_loss_history) >= window:
        gen_ma = np.convolve(gen_loss_history, np.ones(window) / window, mode='valid')
        critic_ma = np.convolve(critic_loss_history, np.ones(window) / window, mode='valid')
        plt.plot(gen_ma, label='Generator loss (MA)', linewidth=2, color='blue')
        plt.plot(critic_ma, label='Critic loss (MA)', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN-GP Training Loss (10-epoch Moving Average)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"wgan_image_at_epoch_{epoch:04d}.png"))
    plt.show()

    print("\nIMPROVED WGAN-GP training completed!")
    print("Check the loss curves - they should be much more stable now!")

    NUM_IMAGES = 10
    TEST_DIR = "/content/wgan2_outputs2"
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

    plt.suptitle("Generated WGAN Test Images", fontsize=14)
    plt.show()

    print(f"🖼️ {NUM_IMAGES} novih slika spremljeno u:\n{TEST_DIR}")
    print("Min pixel:", generated_images.numpy().min())
    print("Max pixel:", generated_images.numpy().max())

    # pokrenuti 3.
    import shutil
    # from google.colab import files

    zip_path = "/content/wgan2_outputs2.zip"

    shutil.make_archive(
        base_name="/content/wgan2_outputs2",
        format="zip",
        root_dir=BASE_OUTPUT_DIR
    )

    files.download(zip_path)