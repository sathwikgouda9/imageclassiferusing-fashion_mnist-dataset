#THIS CODE IS WRITTEN ORIGINALLY IN GOOGLE COLAB SO I WILL DIVE IT CELL TO CELL FOR BETTER UNDERSTANDING
#cell1
import tensorflow_datasets as tfds

dataset, info = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
#cell2
import matplotlib.pyplot as plt

train_dataset = dataset['train']
examples_to_display = train_dataset.take(9)

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(examples_to_display):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(info.features['label'].int2str(label))
    plt.axis('off')
plt.show()
#cell3
import tensorflow as tf
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

dataset['train'] = dataset['train'].map(normalize_img)
dataset['test'] = dataset['test'].map(normalize_img)
#cell 4
import tensorflow as tf

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

dataset['train'] = dataset['train'].map(normalize_img)
dataset['test'] = dataset['test'].map(normalize_img)
#cell5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
#cell6
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
#cell7
BUFFER_SIZE = info.splits['train'].num_examples
BATCH_SIZE = 32

train_batches = dataset['train'].shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = dataset['test'].batch(BATCH_SIZE)

history = model.fit(train_batches, epochs=10, validation_data=test_batches)
