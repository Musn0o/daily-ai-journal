import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # or "tensorflow" or "torch"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
#update