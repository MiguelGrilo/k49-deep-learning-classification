# ================================================================
# Final Project - Artificial Neuronal Networks
# Developed by:
# - Miguel Grilo    |   58387   |   AI & Data Science   |   UÉvora
# - Jorge Couto     |   58656   |   AI & Data Science   |   UÉvora
# ================================================================

import argparse

from functions import load_and_prepare_data, setup_environment, setup_keras_environment, setup_pytorch_environment, save_report
from functions.paths import DATASET_DIR
from versions.c01_simple_ann import iter2 as c01
from versions.c02_deep_ann import iter2 as c02
from versions.c03_cnn import iter5 as c03
from versions.c04_pytorch_ann import iter2 as c04
from versions.c05_ann_ensemble import iter1 as c05
from versions.c06_tl_resnet import iter3 as c06
from versions.c07_keras_tuner import iter2 as c07
from versions.c08_residual_cnn import iter2 as c08
from versions.c09_mixup_cnn import iter2 as c09
from versions.c10_tl_yolov8 import iter4 as c10
from versions.c11_mixed_ensemble import iter1 as c11

MODELS_REGISTRY = {
    "1":  {"name": "Simple ANN",    "runner": c01,                                 "library": "keras"},
    "2":  {"name": "Deep ANN",      "runner": c02,                                 "library": "keras"},
    "3":  {"name": "Advanced CNN",  "runner": c03, "cache_prob": True,             "library": "keras"},
    "4":  {"name": "PyTorch ANN",   "runner": c04,                                 "library": "pytorch"},
    "5":  {"name": "ANN Ensemble",  "runner": c05,                                 "library": "keras"},
    "6":  {"name": "TL ResNet50V2", "runner": c06, "cache_prob": True,             "library": "keras"},
    "7":  {"name": "Keras Tuner",   "runner": c07,                                 "library": "keras"},
    "8":  {"name": "Residual CNN",  "runner": c08,                                 "library": "keras"},
    "9":  {"name": "MixUp CNN",     "runner": c09, "cache_prob": True,             "library": "keras"},
    "10": {"name": "TL YOLOv8",     "runner": c10,                                 "library": "pytorch"},
    "11": {"name": "Mixed Ensemble","runner": c11, "needs_probs": ["3", "6", "9"], "library": "ensemble"}
}

def print_summary(results):
    print("\n# ===============================================================================")
    print("# FINAL RESULTS SUMMARY (OVERALL METRICS)")
    print("# ===============================================================================")

    for key, model_info in MODELS_REGISTRY.items():
        if key in results:
            acc, prec, rec, f1 = results[key]
            print(f"  {key.rjust(2)}. {model_info['name'].ljust(20)}: Acc: {acc * 100:5.2f}% | Prec: {prec * 100:5.2f}% | Rec: {rec * 100:5.2f}% | F1: {f1 * 100:5.2f}%")
        else:
            print(f"  {key.rjust(2)}. {model_info['name'].ljust(20)}: [NOT RUN]")

    print("# ===============================================================================\n")
    print("[INFO] All Detailed Reports and Confusion Matrix have been saved to the 'results/' folder.")


def main():
    parser = argparse.ArgumentParser(description="Run the final classifiers for K49 Dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Specify which model to run (1 to 11) or 'all' (default) to run everything.",
    )
    args = parser.parse_args()

    # DETERMINE MODELS FIRST
    models_to_run = (list(MODELS_REGISTRY.keys()) if args.model.lower() == "all" else [args.model])

    # SMART GPU ROUTING
    needs_keras = False
    needs_pytorch = False

    for key in models_to_run:
        if key in MODELS_REGISTRY:
            lib = MODELS_REGISTRY[key].get("library", "keras")
            if lib in ["keras", "ensemble"]:
                needs_keras = True
            elif lib == "pytorch":
                needs_pytorch = True

    if needs_keras : setup_keras_environment()
    if needs_pytorch : setup_pytorch_environment()

    # CONTINUE NORMAL PIPELINE
    setup_environment()

    x_train, y_train, x_test, y_test = load_and_prepare_data(dataset_dir=DATASET_DIR)
    data_args = (x_train, y_train, x_test, y_test)

    print("\n# ========================================================")
    print(f"# TRAINING FINAL MODELS (Mode: {args.model.upper()})")
    print("# ========================================================\n")

    results = {}
    probs_cache = {"3": None, "6": None, "9": None}

    for key in models_to_run:
        if key not in MODELS_REGISTRY:
            print(f"[!] Invalid model key: {key}")
            continue

        model = MODELS_REGISTRY[key]
        model_name = model["name"].upper()
        safe_model_name = model_name.replace(" ", "_")
        library = model.get("library", "keras")

        print("\n# =============================")
        print(f"# CLASSIFIER {key}: {model_name}")
        print("# =============================")

        if model.get("needs_probs"):
            acc, prec, rec, f1, prob, trained_model = model["runner"](
                *data_args,
                prob_3=probs_cache.get("3"),
                prob_6=probs_cache.get("6"),
                prob_9=probs_cache.get("9"),
            )
        else:
            acc, prec, rec, f1, prob, trained_model = model["runner"](*data_args)
            if key in probs_cache : probs_cache[key] = prob

        save_report(
            y_true=y_test,
            prob=prob,
            model_name=safe_model_name,
            library=library,
            trained_model=trained_model
        )
        results[key] = (acc, prec, rec, f1)

    print_summary(results)

if __name__ == "__main__":
    main()