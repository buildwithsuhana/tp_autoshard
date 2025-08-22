import os
os.environ["KERAS_BACKEND"] = "jax"

import tensorflow as tf
import numpy as np
import keras
import jax

print(f"✅ Keras backend: {keras.config.backend()}")
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ JAX version: {jax.__version__}")
print("-" * 50)

# 1. Create a native TensorFlow Variable
tf_var = tf.Variable([1.0, 2.0, 3.0])
print(f"Original object type: {type(tf_var)}")
print("-" * 50)

# 2. Attempt the conversion using the robust np.array() method
print("Attempting conversion: TF Variable -> NumPy Array -> JAX Array")
try:
    # Step 2a: Force conversion to a NumPy array
    numpy_array = np.array(tf_var)
    print(f"  Intermediate type (after np.array()): {type(numpy_array)}")

    # Step 2b: Convert the NumPy array using keras.ops
    final_tensor = keras.ops.convert_to_tensor(numpy_array)
    print(f"  Final type (after ops.convert_to_tensor): {type(final_tensor)}")

    if "jax" in str(type(final_tensor)):
        print("\n🎉 SUCCESS: The conversion to a JAX array worked correctly!")
    else:
        print("\n❌ FAILURE: The final object is not a JAX array.")

except Exception as e:
    print(f"\n💥 ERROR: The conversion failed with an exception: {e}")