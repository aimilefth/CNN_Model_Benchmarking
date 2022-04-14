import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import subprocess
print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
#tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.threading.set_inter_op_parallelism_threads(4)
print((subprocess.check_output("lscpu | grep 'Model name'", shell=True).strip()).decode())
print((subprocess.check_output("lscpu | grep 'Socket(s):'", shell=True).strip()).decode())
print((subprocess.check_output("lscpu | grep 'Core(s) per socket'", shell=True).strip()).decode())
print((subprocess.check_output("lscpu | grep 'Thread(s) per core'", shell=True).strip()).decode())
#!lscpu | grep 'Model name'
#!lscpu | grep 'Socket(s):'
#!lscpu | grep 'Core(s) per socket'
#!lscpu | grep 'Thread(s) per core'
#!nvidia-smi

def normalize_image(image, label):
    return tf.cast(x=image, dtype=tf.float32)/255.0, label
def rgb_to_gray(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

(ds_test_single), ds_info = tfds.load(
    name="cifar10",
    split="test[0:1000]",
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

AUTOTUNE = tf.data.AUTOTUNE
ds_test_single = ds_test_single.map(map_func=rgb_to_gray, num_parallel_calls=AUTOTUNE)
ds_test_single = ds_test_single.map(map_func=normalize_image, num_parallel_calls=AUTOTUNE)
ds_test_single = ds_test_single.batch(batch_size=1)
ds_test_single = ds_test_single.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model("./LeNet5_Cifar10_47pct_0_5MF")

with tf.device('/cpu:0'):
    model.evaluate(ds_test_single, verbose=1)
