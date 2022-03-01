import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as pyplotMat
import os
import tensorflow_docs.vis.embed as embed
import PIL
import numpy as numpy
from tensorflow.keras import layers
from IPython import display

(to_train, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
to_train = to_train.reshape(to_train.shape[0], 28, 28, 1).astype('float32')
to_train = (to_train - 127.5) / 127.5
size_of_buffer = 60000
size_of_batch = 256
train_dataset = tf.data.Dataset.from_tensor_slices(to_train).shuffle(size_of_buffer).batch(size_of_batch)

def create_gm():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, inumpyut_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model
	
generator = create_gm()
noise = tf.random.normal([1, 100])
image_to_check = generator(noise, training=False)
pyplotMat.imshow(image_to_check[0, :, :, 0], cmap='gray')
def create_dm():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     inumpyut_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model
	
discriminator = create_dm()
decision = discriminator(image_to_check)
print (decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    loss_actual = cross_entropy(tf.ones_like(real_output), real_output)
    loss_wrong_detected = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (loss_actual + loss_wrong_detected)
	
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])
@tf.function
def train_step(images):
    noise = tf.random.normal([size_of_batch, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        image_to_checks = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(image_to_checks, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
	
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        save_files(generator,
                             epoch + 1,
                             seed)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    display.clear_output(wait=True)
    save_files(generator,
                           epochs,
                           seed)
						   
def save_files(model, epoch, test_inumpyut):
    predictions = model(test_inumpyut, training=False)
    fig = pyplotMat.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        pyplotMat.subplot(4, 4, i+1)
        pyplotMat.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        pyplotMat.axis('off')
    pyplotMat.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    pyplotMat.show()
	
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
	
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
	
embed.embed_file(anim_file)	
