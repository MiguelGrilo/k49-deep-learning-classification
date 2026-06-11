import os
import urllib.request

import numpy as np
from functions.paths import DATASET_DIR


# ================
# SETUP GPU MEMORY
# ================
def setup_gpu_memory():
    import os
    os.environ["LD_LIBRARY_PATH"] = "/opt/cuda/lib64:/usr/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    import tensorflow as tf

    """
    Configures TensorFlow to allocate GPU memory dynamically (Memory Growth).
    Allows TensorFlow and other libraries (such as PyTorch) to share VRAM.
    """

    # Memory Growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU SETUP] Memory Growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[GPU SETUP ERROR]: {e}")
    else:
        print("[GPU SETUP] No GPU detected. Training will be performed on the CPU.")


# =====================
# LOAD AND PREPARE DATA
# =====================
def load_and_prepare_data(dataset_dir=DATASET_DIR):
    """
    Downloads, loads, and normalizes the official K49 dataset.
    Arguments:
    - dataset_dir: Directory where the .npz files will be downloaded and stored.
    Returns:
    - x_train, y_train, x_test, y_test: Normalized numpy arrays for training and testing.
    """

    os.makedirs(dataset_dir, exist_ok=True)

    # Helper function to download datasets directly from the official source
    def download_k49(filename):
        base_url = "https://codh.rois.ac.jp/kmnist/dataset/k49/"
        filepath = os.path.join(dataset_dir, filename)

        if not os.path.exists(filepath):
            print(
                f"Downloading {filename} into '{dataset_dir}/' (might take a moment)..."
            )
            urllib.request.urlretrieve(base_url + filename, filepath)
        else:
            print(f"{filepath} already exists. Skipping download.")

    files = [
        "k49-train-imgs.npz",
        "k49-train-labels.npz",
        "k49-test-imgs.npz",
        "k49-test-labels.npz",
    ]
    for f in files : download_k49(f)

    # Load NumPy arrays
    print("\nLoading K49 arrays into memory...")
    x_train = np.load(os.path.join(dataset_dir, "k49-train-imgs.npz"))["arr_0"]
    y_train = np.load(os.path.join(dataset_dir, "k49-train-labels.npz"))["arr_0"]
    x_test = np.load(os.path.join(dataset_dir, "k49-test-imgs.npz"))["arr_0"]
    y_test = np.load(os.path.join(dataset_dir, "k49-test-labels.npz"))["arr_0"]

    # Normalization (0 to 1 scale)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(f"Train shape: {x_train.shape} | Labels: {y_train.shape}")
    print(f"Test shape:  {x_test.shape} | Labels: {y_test.shape}\n")

    return x_train, y_train, x_test, y_test