import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from functions.paths import RESULTS_DIR, WEIGHTS_DIR


def setup_environment():
    """Ensures necessary directories exist."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def setup_keras_environment():
    """
    Initializes TensorFlow/Keras and configures dynamic memory growth.
    This prevents TensorFlow from hoarding 100% of the GPU VRAM instantly.
    """

    import os
    os.environ["LD_LIBRARY_PATH"] = "/opt/cuda/lib64:/usr/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    import tensorflow as tf

    print("\n[GPU SETUP] Initializing TensorFlow/Keras Environment...")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU SETUP] Memory Growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[GPU SETUP ERROR]: {e}")
    else:
        print("[GPU SETUP] No GPU detected. TensorFlow will run on CPU.")

def setup_pytorch_environment():
    """
    Initializes PyTorch, clears residual VRAM cache, and scrubs conflicting
    Linux system paths to prevent library mismatch crashes.
    """
    import os

    print("\n[GPU SETUP] Initializing PyTorch Environment...")

    if "LD_LIBRARY_PATH" in os.environ:
        del os.environ["LD_LIBRARY_PATH"]
        print("[GPU SETUP] Scrubbed LD_LIBRARY_PATH to prevent cuDNN conflicts.")

    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

    # Now it is safe to import torch
    import torch

    if torch.cuda.is_available():
        # Clear any ghost memory before starting
        torch.cuda.empty_cache()

        # Disable benchmarking to prevent aggressive algorithm scanning
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(
            f"[GPU SETUP] PyTorch mapped successfully to: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("[GPU SETUP] No GPU detected. PyTorch will run on CPU.")


def save_report(
    y_true,
    prob,
    model_name,
    library,
    trained_model=None,
    output_dir=RESULTS_DIR,
    weights_dir=WEIGHTS_DIR,
):
    """
    Saves the trained model, caches the prediction probabilities,
    converts probabilities into final classes, and generates evaluation reports.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # ==================
    # SAVE MODEL WEIGHTS
    # ==================
    if trained_model is not None:
        if library == "pytorch":
            import torch  # Lazy Import
            model_path = os.path.join(weights_dir, f"{model_name}.pth")

            if hasattr(trained_model, "model"):
                torch.save(trained_model.model.state_dict(), model_path)
            else:
                torch.save(trained_model.state_dict(), model_path)

            print(f"   [+] PyTorch model weights saved to: '{model_path}'")

        elif library == "keras":
            model_path = os.path.join(weights_dir, f"{model_name}.keras")
            trained_model.save(model_path)
            print(f"   [+] Keras model saved to: '{model_path}'")
        else:
            print(f"   [!] Warning: Unrecognized library '{library}'. Could not save '{model_name}'.")

    # ===================
    # CACHE PROBABILITIES
    # ===================
    if prob is not None:
        prob_path = os.path.join(weights_dir, f"{model_name}_prob.npy")
        np.save(prob_path, prob)
        print(f"   [+] Probabilities cached to: '{prob_path}'")

    # =====================================
    # GENERATE REPORTS & CONFUSION MATRICES
    # =====================================
    # Process Labels and Predictions
    y_true_classes = np.argmax(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
    y_pred_classes = np.argmax(prob, axis=-1)

    # Generate and Save Classification Report
    report = classification_report(y_true_classes, y_pred_classes, digits=4)
    report_path = os.path.join(output_dir, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"CLASSIFICATION REPORT: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Save Heatmap (Trends)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix (Trends) - {model_name}", fontsize=16, pad=20)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.tight_layout()
    cm_heatmap_path = os.path.join(output_dir, f"{model_name}_hm.png")
    plt.savefig(cm_heatmap_path, dpi=300)
    plt.close()

    # Save Confusion Matrix (Detailed)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, annot_kws={"size": 6})
    plt.title(f"Confusion Matrix (Detailed) - {model_name}", fontsize=20, pad=20)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("Actual Class", fontsize=14)
    plt.tight_layout()
    cm_annotated_path = os.path.join(output_dir, f"{model_name}_cm.png")
    plt.savefig(cm_annotated_path, dpi=400)
    plt.close()

    print(f"   [+] Reports saved for {model_name} (1 TXT, 2 Images, 1 NPY Cache) in '{output_dir}'")