import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#!lscpu | grep 'Model name'
#!lscpu | grep 'Socket(s):'
#!lscpu | grep 'Core(s) per socket'
#!lscpu | grep 'Thread(s) per core'
#!nvidia-smi

def normalize_image(image, label):
    return tf.cast(x=image, dtype=tf.float32)/255.0, label
def resize_image(image, label):
    new_height = new_width = 224
    image = tf.image.resize(image, (new_height,new_width))
    return image, label

(ds_test_single), ds_info = tfds.load(
    name="cifar10",
    split="test[0:100]",
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
AUTOTUNE = tf.data.AUTOTUNE
ds_test_single = ds_test_single.map(map_func=normalize_image, num_parallel_calls=AUTOTUNE)
ds_test_single = ds_test_single.batch(batch_size=1)
ds_test_single = ds_test_single.prefetch(buffer_size=AUTOTUNE)
ds_test_single = ds_test_single.map(map_func=resize_image, num_parallel_calls=AUTOTUNE)

model = tf.keras.models.load_model("./LeNet5_Cifar10_47pct_0_5MF")

with tf.device('/cpu:0'):
    model.evaluate(ds_test_single, verbose=1)
