import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check if the MPS backend is available
if tf.config.list_physical_devices('GPU'):
    print("MPS backend is available.")
else:
    print("MPS backend is not available. Please ensure you have the correct setup.")
