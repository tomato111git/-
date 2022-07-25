import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape, LeakyReLU
from keras.models import Sequential
img_shape = (28, 28, 1)
z_dim = 100

def build_generator(img_shape,z_dim):
  model = Sequential()
  model.add(Dense(128, input_dim = z_dim))
  model.add(LeakyReLU(alpha = 0.01))
  model.add(Dense(28*28*1, activation="tanh"))
  model.add(Reshape(img_shape))
  return model

def build_descriminator(img_shape):
  model = Sequential()
  model.add(Flatten(input_shape=img_shape))
  model.add(Dense(128))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(1, activation='sigmoid'))
  return model

def build_gan(gen, dis):
  model = Sequential()
  model.add(gen)
  model.add(dis)
  return model


discriminator = build_descriminator(img_shape)
discriminator.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
discriminator.trainable = False

generator = build_generator(img_shape, z_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer="adam")

batch_size = 128

(X_train, _), (_, _) = mnist.load_data()

X_train = X_train/127.5-1.0
X_train = np.expand_dims(X_train, axis=3)

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for i in range(10000):
  idx = np.random.randint(0, X_train.shape[0], batch_size)
  imgs = X_train[idx]
  z = np.random.normal(0, 1, (batch_size, 100))
  fake_imgs = generator.predict(z)

  discriminator.train_on_batch(imgs, real)
  discriminator.train_on_batch(fake_imgs, fake)

  gan.train_on_batch(z, real)

  if i%500 == 0:
    noise = np.random.normal(0, 1, (1, z_dim))
    sample_img = generator.predict(noise)
    sample_img = sample_img*0.5 + 0.5

    plt.imshow(sample_img[0, :, :, 0], cmap="gray")
    plt.show()
