# K49 Character Classification - Deep Learning Models

> This project explores 11 different deep learning architectures to classify the 49 complex Japanese characters of the K49 dataset. It progresses from baseline Artificial Neural Networks to State-of-the-Art Transfer Learning pipelines, ultimately achieving **97.72% Accuracy** using a fine-tuned YOLOv8 architecture.

## Top Performing Models

Out of the 11 architectures explored in this project, the following three achieved State-of-the-Art (SOTA) performance on the K49 dataset:

|  Rank  | Architecture          | Approach                                       |  Accuracy  |  F1-Score  |
|:------:|:----------------------|:-----------------------------------------------|:----------:|:----------:|
| **#1** | **YOLOv8-cls (Nano)** | Transfer Learning (224px, 50 Epochs)           | **97.72%** | **97.85%** |
| **#2** | **Mixed Ensemble**    | Averaged Softmax (CNN + ResNet + MixUp)        | **96.42%** | **96.60%** |
| **#3** | **Advanced CNN**      | Built from scratch (150 Epochs + Augmentation) | **94.27%** | **94.46%** |

*For a detailed breakdown of the hyperparameter tuning, ablation studies, and architectural evolution of all 11 classifiers, please see the [Models & Evaluation Summaries](#models--evaluation-summaries) section below.*

## Table of Contents
1. [Top Performing Models](#top-performing-models)
2. [Dataset Resources](#dataset-resources)
3. [Environment Setup & Installation](#environment-setup--installation)
4. [Project Architecture](#project-architecture)
5. [Hardware Specifications](#hardware-specifications)
6. [Technical Challenges & Computational Constraints](#technical-challenges--computational-constraints)
7. [Models & Evaluation Summaries](#models--evaluation-summaries)
   * [Classifier 1: Simple ANN](#classifier-1-simple-ann)
   * [Classifier 2: Deep ANN](#classifier-2-deep-ann)
   * [Classifier 3: Advanced CNN](#classifier-3-advanced-cnn)
   * [Classifier 4: PyTorch ANN](#classifier-4-pytorch-ann)
   * [Classifier 5: Ensemble Learning](#classifier-5-ann-ensemble-learning)
   * [Classifier 6: Transfer Learning (ResNet50V2)](#classifier-6-transfer-learning-resnet50v2)
   * [Classifier 7: Hyperparameter Tuning (KerasTuner)](#classifier-7-keras-tuner)
   * [Classifier 8: Residual CNN (Skip Connections)](#classifier-8-residual-cnn-skip-connections)
   * [Classifier 9: CNN with MixUp Augmentation](#classifier-9-cnn-with-mixup-augmentation)
   * [Classifier 10: Transfer Learning (YOLOv8-cls)](#classifier-10-transfer-learning-yolov8-cls)
8. [Conclusions](#conclusions)

---

## Dataset Resources
* [PyTorch Vision Datasets Main Page](https://docs.pytorch.org/vision/main/datasets.html)
* [KMNIST / K49 Dataset Official Documentation](https://datasets-dev.10web.me/docs/ml/datasets/kmnist/)
* [KMNIST / K49 Dataset GitHub Repository](https://github.com/rois-codh/kmnist)

---

## Environment Setup & Installation

This project utilizes [uv](https://docs.astral.sh/uv/) for lightning-fast and reproducible Python package management. The dependencies are locked in the `uv.lock` file to guarantee that all environments are identical.

### 1. Prerequisites
Ensure you have Python 3.11.9 or higher installed on your system.

### 2. Install UV
If you don't have `uv` installed, you can install it globally via pip (or check their [official documentation](https://docs.astral.sh/uv/getting-started/installation/) for other methods):
```bash
pip install uv
```

### 3. Create a Virtual Environment and Install Dependencies
Navigate to the root directory of the project. We provide specific installation paths depending on your hardware and Operating System. This ensures that users without an NVIDIA GPU do not have to download gigabytes of unnecessary CUDA drivers.

**Option A: CPU Only / macOS / AMD**

Run this command if you do not have a dedicated NVIDIA GPU, or if you are on a Mac. It will automatically create a `.venv` and install all base packages:
```bash
uv sync
```

**Option B: NVIDIA GPU (Linux: Ubuntu, Debian, Arch)**

Run this command to install the heavy CUDA drivers and automatically map the C++ paths required for TensorFlow hardware acceleration via an `.env` file:
```bash
uv sync --extra cuda
# Creates .env with system/venv CUDA paths
echo "LD_LIBRARY_PATH=$(find "$PWD/.venv" -type d -path '*/nvidia/*/lib' 2>/dev/null | paste -sd ':' -):/usr/lib:/usr/local/cuda/lib64:/opt/cuda/lib64" > .env
```

**Option C: NVIDIA GPU (Windows via WSL2)**

For Windows users running the project through WSL2:
```bash
uv sync --extra cuda
# Creates .env with system/venv CUDA paths
echo "LD_LIBRARY_PATH=$(find "$PWD/.venv" -type d -path '*/nvidia/*/lib' 2>/dev/null | paste -sd ':' -):/usr/lib/wsl/lib" > .env
```

### (Optional) 4. Activate the Virtual Environment
The `uv run` command automatically detects and uses the virtual environment. However, if you wish to activate it manually for your terminal session:
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

### 5. Run the Project
With the environment set up, you can now run the main execution pipeline. 

*If you are using a GPU (Options B or C), you must pass the generated `.env` file to enable hardware acceleration. Thanks to our internal environment router, this single command safely manages VRAM allocation for both TensorFlow and PyTorch sequentially:*
```bash
uv run --env-file .env python3 main.py --model all
```
*(Note: CPU/macOS users can simply run `uv run python3 main.py --model all`)*

### 6. Run Individual Model Versions (Ablation Studies)
While `main.py` serves as the central executive pipeline utilizing cached weights for rapid evaluation, the `versions/` directory contains the standalone scripts used during the actual development and hyperparameter tuning. 

To train a specific classifier from scratch or observe the epoch-by-epoch training logs, replace `<filename>` with the desired script (e.g., `c01_simple_ann`) and select the iteration:

**Using the specific version script:**
```bash
# For GPU users (The .env file works safely for ALL models, both Keras and PyTorch)
uv run --env-file .env python3 -m versions.<filename> --iter <iter_number>

# For CPU users:
uv run python3 -m versions.<filename> --iter <iter_number>
```

**Using the main orchestrator for a specific model (Latest versions):**
```bash
# For GPU users:
uv run --env-file .env python3 main.py --model <model_number>

# For CPU users:
uv run python3 main.py --model <model_number>
```

---

## Project Architecture

```plaintext
FinalProject_58387_58656/
├── dataset/
│   ├── yolo_format/
│   ├── k49-test-imgs.npz
│   ├── k49-test-labels.npz
│   ├── k49-train-imgs.npz
│   └── k49-train-labels.npz
│
├── functions/
│   ├── __init__.py
│   ├── ann.py (simple_ann, deep_ann, pytorch_ann)
│   ├── cnn.py (advanced_cnn, residual_cnn, mixup_cnn)
│   ├── ensemble.py (evaluate_ensemble)
│   ├── keras_training.py (keras_classifier)
│   ├── keras_tuner.py (keras_tuner)
│   ├── main_utils.py (MODELS_REGISTRY, setup_environment, print_summary)
│   ├── paths.py (Used DIRS)
│   ├── transfer_learning.py (tl_resnet, tl_yolo)
│   └── utils.py (setup_gpu_memory, load_and_prepare_data, save_report)
│
├── keras_tuner/
│
├── results
│   ├── c01_simple_ann_cm.png
│   ├── c01_simple_ann_hm.png
│   ├── c01_simple_ann_report.txt
│   ├── ...
│   └── c11_mixed_ensemble_report.txt
│
├── runs/classify/YOLO_K49_Runs/
│
├── versions/
│   ├── c01_simple_ann.py
│   ├── ...
│   └── c11_mixed_ensemble.py
│
├── weights/
│   ├── c01_simple_ann.keras
│   ├── ...
│   └── yolov8n-cls.pt
│
├── .env
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

### Directory Breakdown

* **`main.py`**: The central execution pipeline. It acts as an executive summary, orchestrating the loading of data, model training/caching, and final metric generation without cluttering the view with raw architecture code.
* **`dataset/`**: Contains the raw K49 `.npz` files as well as the auto-generated `yolo_format/` directory required for State-of-the-Art transfer learning pipelines.
* **`functions/`**: The core Python package of the project, cleanly exposing all internal modules (like `keras_training.py` for robust GPU memory management) to the main pipeline.
* **`versions/`**: An archive of standalone scripts detailing the exact iterative hyperparameter tuning processes and ablation studies that led to the final architectures.
* **`keras_tuner/` & `runs/`**: Auto-generated directories. `keras_tuner/` stores the exploration states of hyperparameter optimization, while `runs/` stores the Ultralytics YOLOv8 training logs and validation metrics.
* **`weights/` & `results/`**: Auto-generated directories storing the serialized model checkpoints (`.keras`, `.pt`), predicted probabilities (`.npy`), and exported confusion matrices/classification reports.

---

## Hardware Specifications
All models in this repository were trained and evaluated locally using an **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**. Developing heavy architectural pipelines under these hardware constraints required highly specialized memory management algorithms and structural workarounds (detailed in the next section).

---

## Technical Challenges & Computational Constraints

Developing and training 11 distinct deep learning architectures on a local machine presented significant software engineering and hardware optimization hurdles.

### 1. The Computational Cost of Deep Architectures
Training State-of-the-Art visual models is incredibly resource-intensive. The **YOLOv8-cls** architecture, for example, required upscaling the K49 dataset images from their native 28x28 pixels to 224x224 pixels so the deep residual layers could properly extract spatial features. Running this configuration across 50 epochs required **hours of uninterrupted GPU computing**. Similarly, exploring the vast parameter space using `keras-tuner` and training the **Advanced CNN** from scratch for 150 epochs pushed the RTX 4050 to its absolute limits, requiring dynamic batch sizing and early-stopping mechanisms to make the project viable locally.

### 2. The PyTorch vs. TensorFlow "VRAM War"
The most complex engineering challenge involved managing the Linux (Arch) environment variables and C++ libraries. Both TensorFlow and PyTorch are inherently aggressive regarding GPU allocation—by default, they attempt to hijack 100% of the available GPU VRAM upon initialization. Furthermore, they rely on conflicting versions of `cuDNN` mapping. 
To successfully run a pipeline that seamlessly orchestrates both Keras/TensorFlow models and PyTorch/Ultralytics models, we had to implement a strict environment firewall:
* **TensorFlow Isolation:** For Keras models, we utilized a `.env` file to map the system's `LD_LIBRARY_PATH` explicitly to the CUDA binaries, alongside a custom Python wrapper (`setup_keras_environment()`) that forces `tf.config.experimental.set_memory_growth` to allow dynamic RAM allocation rather than instantaneous overallocation.
* **PyTorch / YOLO Isolation:** PyTorch models required the exact opposite approach. Because PyTorch on Linux is highly prone to segmentation faults when colliding with system-level `cuDNN` libraries, we engineered a "Nuclear Pre-Boot Fix" (`os.environ` deletion of `LD_LIBRARY_PATH`) deployed at the top of the execution scripts. This purposefully blinds PyTorch to the system OS, forcing it to securely use its own pip-installed wheels, thus preventing immediate system crashes during the YOLO execution.

---

## Models & Evaluation Summaries

### Classifier 1: Simple ANN
#### Description
A foundational Artificial Neural Network (ANN) used as the starting point for the K49 character classification. This model serves to establish the baseline performance metrics before introducing more complex architectures, regularization techniques, or deep layers.

#### Versions
* **Iteration 1 (Baseline):** The default implementation of the simple ANN. It runs with the base parameters (128 hidden neurons and 5 epochs) to map the initial learning capability of a basic fully connected network.
* **Iteration 2 (Optimized):** A scaled-up version aiming to improve the baseline. The network's capacity was expanded by increasing the `hidden_neurons` from 128 to 256, and the training duration was extended by bumping the `epochs` from 5 to 15, using a `batch_size` of 256. 

#### Results
|     Iteration     |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-----------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Baseline)**  |   74.25%   |   75.51%   |   74.25%   |   74.56%   |
| **2 (Optimized)** | **81.43%** | **82.03%** | **81.43%** | **81.53%** |


### Classifier 2: Deep ANN
#### Description
A multi-layer Deep Artificial Neural Network designed to capture more complex, non-linear relationships within the K49 dataset. By stacking multiple hidden layers, this architecture allows the model to learn deeper hierarchical feature representations of the handwritten characters compared to the simple ANN.

#### Versions
* **Iteration 1 (Baseline):** The default implementation featuring a standard deep architecture (using 300 neurons in the first hidden layer and 100 in the second) to establish an initial performance baseline for multi-layer networks.
* **Iteration 2 (Optimized):** An upgraded architecture with expanded computational capacity. The dense layers were significantly widened (increased to 512 and 128 neurons) to provide the network with more parameters, allowing it to better map and differentiate the 49 complex Japanese characters.

#### Results
|     Iteration     |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-----------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Baseline)**  |   81.93%   |   82.49%   |   81.93%   |   82.07%   |
| **2 (Optimized)** | **84.90%** | **85.80%** | **84.90%** | **85.13%** |


### Classifier 3: Advanced CNN
#### Description
A Convolutional Neural Network (CNN) specifically tailored for spatial feature extraction in image classification. This architecture evolves progressively from a baseline model to an advanced, highly robust configuration by incorporating deeper convolutional blocks, data augmentation, batch normalization, L2 regularization, and dynamic learning rate scheduling to maximize generalization on the K49 dataset.

#### Versions
* **Iteration 1 (Baseline):** The default CNN implementation serving as the baseline for spatial feature extraction.
* **Iteration 2 (Dense=256):** Maintained the core convolutional architecture but increased the final dense layer to 256 neurons to allow for better feature combination before the final classification.
* **Iteration 3 (Deep+Aug+Reg):** A structural overhaul that added depth (3 Convolutional blocks), Data Augmentation (0.05 rotation), Batch Normalization, L2 Regularization, and Dropout (0.4). Trained for 40 epochs with early stopping patience of 5.
* **Iteration 4 (LR Scheduler):** Refined the augmentation by reducing rotation to 0.03, increased early stopping patience to 8, and activated a Dynamic Learning Rate (ReduceLROnPlateau) over 50 epochs to help the model converge smoother.
* **Iteration 5 (150 Epochs):** Pushed the limits of the architecture by extending the training to 150 epochs, allowing the Data Augmentation strategy to fully generalize and reach state-of-the-art performance for this network size.

#### Results
|      Iteration       |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:--------------------:|:----------:|:----------:|:----------:|:----------:|
|   **1 (Baseline)**   |   90.56%   |   90.57%   |   90.56%   |   90.48%   |
|  **2 (Dense=256)**   |   91.71%   |   92.09%   |   91.71%   |   91.81%   |
| **3 (Deep+Aug+Reg)** |   90.88%   |   91.42%   |   90.88%   |   90.99%   |
| **4 (LR Scheduler)** |   93.84%   |   94.59%   |   93.84%   |   94.15%   |
|  **5 (150 Epochs)**  | **94.27%** | **94.79%** | **94.27%** | **94.46%** |


### Classifier 4: PyTorch ANN
#### Description
A PyTorch-based implementation of a simple Artificial Neural Network. This classifier serves to demonstrate the flexibility of building custom training loops and network architectures outside the TensorFlow/Keras ecosystem, acting as a framework comparison for classifying the K49 dataset.

#### Versions
* **Iteration 1 (Baseline):** The default PyTorch implementation utilizing Stochastic Gradient Descent (SGD) and a basic network capacity to establish a baseline performance.
* **Iteration 2 (Adam Opt):** An optimized version where the SGD optimizer was replaced with the more efficient Adam optimizer (Learning Rate = 0.001). The network's capacity was increased to 256 hidden neurons, and training was extended to 15 epochs to match the configuration of the previous Keras models.

#### Results
|    Iteration     |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:----------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Baseline)** |   72.63%   |   74.52%   |   72.63%   |   73.20%   |
| **2 (Adam Opt)** | **80.66%** | **81.17%** | **80.66%** | **80.71%** |


### Classifier 5: ANN Ensemble Learning
#### Description
An Ensemble Learning approach that combines the predictive power of multiple independent Deep Artificial Neural Networks. By averaging the Softmax probability distributions of three diverse models, this technique leverages the "wisdom of the crowd" to reduce prediction variance, mitigate individual model biases, and improve overall generalization on the K49 dataset.

#### Versions
Instead of iterative improvements, this classifier involves training three distinct "member" models concurrently (Phase 1) and aggregating their predictions (Phase 2):
* **Member A (Standard):** A Deep ANN utilizing standard optimal parameters (512 → 128 neurons) trained for 30 epochs.
* **Member B (High Dropout):** Shares the same neuron capacity as Member A, but utilizes a higher `dropout_rate` (0.4) to deliberately force the network to learn different, more robust feature representations.
* **Member C (Structural Variation):** Introduces architectural diversity by altering the layer capacities (400 → 200 neurons), ensuring the ensemble members make distinct types of errors.
* **Final Ensemble:** The ultimate classifier that averages the output probabilities of Models A, B, and C to yield a single, highly stable prediction.

#### Results
*Averaging Softmax probabilities of 3 different Deep ANN architectures.*

| Member / Ensemble  |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:------------------:|:----------:|:----------:|:----------:|:----------:|
| Member A (Indiv.)  |   84.36%   |     -      |     -      |   84.65%   |
| Member B (Indiv.)  |   83.18%   |     -      |     -      |   83.43%   |
| Member C (Indiv.)  |   84.74%   |     -      |     -      |   84.92%   |
| **FINAL ENSEMBLE** | **85.91%** | **87.01%** | **85.91%** | **86.27%** |


### Classifier 6: Transfer Learning (ResNet50V2)
#### Description
A Transfer Learning approach leveraging the powerful, pre-trained **ResNet50V2** architecture (originally trained on the ImageNet dataset). This classifier adapts a deep vision model to the K49 dataset, demonstrating the dramatic impact of unlocking and fine-tuning pre-trained weights to transition from recognizing general objects (like animals or cars) to interpreting specific Japanese handwritten characters.

#### Versions
* **Iteration 1 (Frozen Base):** The baseline implementation where the pre-trained ResNet50V2 base weights are completely frozen. Only the newly added classification head is trained, resulting in heavily bottlenecked performance due to the domain shift between ImageNet and K49.
* **Iteration 2 (Fine-Tuning):** Introduces a two-phase training strategy. After an initial warmup period (10 epochs), the base model is unfrozen and fine-tuned for 15 additional epochs with a microscopic learning rate (`1e-5`). This allows the network to adapt its internal feature maps directly to the K49 characters, resulting in a massive accuracy jump.
* **Iteration 3 (SOTA Setup):** Pushes the architecture to State-of-the-Art (SOTA) performance by addressing structural limitations. Images are upscaled to 72x72 to provide more resolution for the deep spatial filters. It also introduces Data Augmentation to prevent overfitting and switches to Global MaxPooling (which is highly effective at detecting sharp, black-and-white character strokes). The fine-tuning phase is extended to 20 epochs to properly adjust the higher-resolution features.

#### Results
|      Iteration      |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-------------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Frozen Base)** |   54.87%   |   57.00%   |   54.87%   |   55.44%   |
| **2 (Fine-Tuning)** |   54.31%   |   60.09%   |   54.31%   |   54.90%   |
| **3 (SOTA Setup)**  | **93.07%** | **93.71%** | **93.07%** | **93.30%** |


### Classifier 7: Keras Tuner
#### Description
An automated Hyperparameter Optimization (HPO) approach using the `keras-tuner` library. Instead of manually guessing the best architecture, this classifier dynamically searches through a predefined hyperparameter space (adjusting convolutional filters, dense units, dropout rates, and learning rates) to algorithmically discover the most effective structural configuration for the K49 dataset.

#### Versions
* **Iteration 1 (Fast Search):** A rapid exploratory search testing 3 different hyperparameter combinations, running for 5 epochs each. Designed to quickly identify a strong baseline configuration without massive computational overhead.
* **Iteration 2 (Deep Search):** A much more exhaustive search testing 10 different combinations, running for 12 epochs each. This iteration aims to squeeze out the absolute maximum performance by thoroughly exploring the hyperparameter grid.

#### Trials, Parameters Tested & Conclusions
During the search processes, the KerasTuner explored the following parameter space:
* **Conv2D Filters:** 32, 64, 96, 128
* **Dense Units:** 128, 256, 384, 512
* **Dropout Rate:** 0.2, 0.3, 0.4
* **Learning Rate (Adam):** 0.01, 0.001, 0.0001

**Key Conclusions:** An interesting Machine Learning phenomenon occurred during this experiment. The *Fast Search* found an optimal configuration utilizing **64 filters, 512 dense units, 0.4 dropout, and a 0.001 learning rate**. The *Deep Search*, despite evaluating more models for longer periods, selected a model with less regularization (**0.2 dropout**) and a slower learning rate (**0.0001**). 

Because the Deep Search model had lower dropout, it slightly overfits the validation data during the tuning process. Consequently, when evaluated on the unseen Test dataset, the simpler "Fast Search" actually generalized better and achieved a marginally higher final accuracy (89.89% vs 89.69%). This perfectly demonstrates that aggressive regularization (higher dropout) is crucial for real-world generalization on complex datasets like K49.

#### Results
|      Iteration      |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-------------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Fast Search)** |   89.79%   |   89.89%   |   89.79%   |   89.71%   |
| **2 (Deep Search)** | **90.10%** | **90.67%** | **90.10%** | **90.28%** |


### Classifier 8: Residual CNN (Skip Connections)
#### Description
A Convolutional Neural Network built using the Keras Functional API to implement **Residual Blocks** (skip connections). By adding the original input of a block to its output before the activation function, this architecture mitigates the vanishing gradient problem, allowing the network to learn deeper and more complex spatial features without degrading.

#### Versions
* **Iteration 1 (Baseline):** The initial implementation of the Residual Block architecture. Trained for 20 epochs with a batch size of 256 to establish the baseline performance of the skip connections.
* **Iteration 2 (40 Epochs):** An extended training phase. The epochs were doubled to 40 under the hypothesis that residual networks excel with longer optimization periods. Interestingly, the performance *dropped* compared to the baseline, suggesting that without heavier regularization (like increased dropout or data augmentation), the model began to overfit the training data.

#### Results
|     Iteration     |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-----------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (Baseline)**  |   89.27%   |   89.96%   |   89.27%   |   89.49%   |
| **2 (40 Epochs)** | **90.54%** | **90.81%** | **90.54%** | **90.56%** |


### Classifier 9: CNN with MixUp Augmentation
#### Description
This model implements a Convolutional Neural Network (CNN) utilizing **MixUp Data Augmentation** during the training phase. MixUp acts as a highly effective regularization technique by simultaneously blending pairs of images and their respective labels. This forces the model to generalize better and considerably reduces the likelihood of overfitting by learning linear combinations of features.

#### Versions
* **Iteration 1 (40 Epochs):** The baseline implementation of the CNN trained with MixUp Data Augmentation. Trained for 40 epochs with a batch size of 256.
* **Iteration 2 (80 Epochs):** An extended training phase. Because MixUp introduces strong regularization (making the training data significantly more challenging and varied), the network requires more epochs to fully converge and extract the maximum benefit from the technique. Training was doubled to 80 epochs.

#### Results
|     Iteration     |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:-----------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (40 Epochs)** |   91.33%   |   92.29%   |   91.33%   |   91.68%   |
| **2 (80 Epochs)** | **91.35%** | **92.29%** | **91.35%** | **91.70%** |


### Classifier 10: Transfer Learning (YOLOv8-cls)
#### Description
A State-of-the-Art (SOTA) Transfer Learning approach using the Ultralytics **YOLOv8-cls** (Nano) architecture, specifically adapted for image classification. Unlike standard Keras models that read arrays directly from RAM, this implementation automatically exports the K49 dataset into a YOLO-compatible folder structure on disk. YOLOv8 leverages highly advanced internal data augmentation and deep convolutional layers to achieve exceptional generalization.

#### Versions
* **Iteration 1 (10 Epochs, 32px):** The baseline YOLOv8 Nano implementation. Trained for 10 epochs with a batch size of 256 using 32x32 pixel images (scaled internally by YOLO).
* **Iteration 2 (25 Epochs, 32px):** Extended the training duration to 25 epochs. This longer cycle allows Ultralytics' aggressive internal data augmentation engine to expose the model to more structural variations.
* **Iteration 3 (15 Epochs, 128px):** A critical architectural fix that unlocked SOTA performance. Because YOLOv8 is a very deep network with multiple downsampling operations, a 32x32 input shrinks to a 1x1 pixel feature map at the deepest layers, destroying the spatial structure of the characters. By drastically upscaling the input resolution to 128x128 (and reducing batch size to 128 to save memory), the deep layers finally have sufficient spatial features to extract, resulting in a massive accuracy spike.
* **Iteration 4 (50 Epochs, 224px):** The "Overkill Mode". This iteration upscales the images to 224x224, matching the exact native resolution YOLOv8 was originally pre-trained on (ImageNet). Training was extended to 50 epochs to allow the weights to fully stabilize, and the batch size was reduced to 64 to prevent Out of Memory (OOM) errors on the GPU. This configuration squeezed out the absolute maximum performance possible, pushing the F1-Score past 97%.

#### Results
|        Iteration         |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:------------------------:|:----------:|:----------:|:----------:|:----------:|
| **1 (10 Epochs, 32px)**  |   80.48%   |   81.70%   |   80.48%   |   80.69%   |
| **2 (25 Epochs, 32px)**  |   92.31%   |   93.18%   |   92.31%   |   92.57%   |
| **3 (15 Epochs, 128px)** |   96.23%   |   96.56%   |   96.23%   |   96.37%   |
| **4 (50 Epochs, 224px)** | **97.72%** | **97.80%** | **97.72%** | **97.85%** |


### Classifier 11: Mixed Ensemble Learning
#### Description
A highly robust Mixed Ensemble Learning approach that combines the predictive power of three drastically different neural network architectures: an Advanced CNN (trained from scratch), a Transfer Learning ResNet50V2 (pre-trained on ImageNet), and a CNN trained with MixUp Data Augmentation. Because these models possess fundamentally different structures and training methodologies, they make different types of errors. By averaging their Softmax probabilities, the ensemble effectively cancels out individual biases, achieving maximum generalization on the K49 dataset.

#### Versions
Instead of iterative epochs, this classifier operates in two distinct phases to combine previously discovered optimal configurations:
* **Phase 1 (Member Initialization):** The ensemble trains the best versions of three independent models: the Advanced CNN (150 epochs), the ResNet50V2 (SOTA setup with 72x72 upscaling), and the MixUp CNN (80 epochs).
* **Phase 2 (Dynamic Averaging):** Test images are dynamically processed to meet the specific input requirements of each model (e.g., resizing to 72x72 and duplicating channels to RGB for the ResNet50V2). The final prediction is generated by averaging the Softmax probability distributions of all three models and selecting the class with the highest combined confidence.

#### Results
*Averaging Softmax probabilities of 3 diverse, State-of-the-Art network architectures.*

| Member / Ensemble  |  Accuracy  | Precision  |   Recall   |  F1-Score  |
|:------------------:|:----------:|:----------:|:----------:|:----------:|
| Member 1 (Adv CNN) |   94.27%   |   94.79%   |   94.27%   |   94.46%   |
| Member 2 (ResNet)  |   93.07%   |   93.71%   |   93.07%   |   93.30%   |
|  Member 3 (MixUp)  |   91.35%   |   92.29%   |   91.35%   |   91.70%   |
| **FINAL ENSEMBLE** | **96.42%** | **96.50%** | **96.42%** | **96.60%** |

<br>

## Conclusions

This project successfully demonstrates the evolution of deep learning architectures applied to complex character recognition, scaling from baseline models to State-of-the-Art transfer learning pipelines. 

Through rigorous ablation studies and hyperparameter tuning, several key insights were discovered:
1. **Spatial Awareness is Critical:** Moving from flat Artificial Neural Networks (ANNs) to Convolutional Neural Networks (CNNs) provided an immediate, massive leap in performance, proving that preserving the 2D spatial hierarchy of the Japanese characters is essential.
2. **The Resolution Bottleneck in Transfer Learning:** When adapting deep, pre-trained vision models like ResNet50V2 and YOLOv8, the native 28x28 resolution of the K49 dataset caused severe feature degradation in the deeper layers. Upscaling the inputs up to 224x224 for YOLOv8 was the definitive factor that unlocked SOTA performance (97.72%).
3. **Regularization Beats Capacity:** As seen in the KerasTuner experiments and the Residual CNN trials, simply adding more layers or neurons quickly led to overfitting. Advanced regularization techniques like **MixUp Data Augmentation** and aggressive Dropout rates were far more effective at improving real-world generalization on unseen test data.
4. **The Power of Ensembles:** The Mixed Ensemble model proved that averaging the predictive distributions of fundamentally different architectures (a scratch-built CNN, a MixUp CNN, and a pre-trained ResNet) successfully mitigates individual network biases, resulting in a highly robust 96.42% accuracy without needing the massive computational overhead of YOLOv8.
5. **Ensemble Compatibility and Model Dominance:** The YOLOv8 architecture was intentionally excluded from the Mixed Ensemble for two engineering reasons. First, its underlying framework (Ultralytics) and disk-based preprocessing pipeline are structurally incompatible with the dynamic, RAM-based prediction averaging used by the Keras models. Second, because YOLOv8's standalone accuracy (97.72%) significantly outpaced the other networks, mathematically averaging its probabilities with weaker classifiers would have diluted its predictions and actively dragged its score down.

**Final Verdict:** For absolute maximum accuracy, the **fine-tuned YOLOv8-cls** is the superior model. However, for a balance of high accuracy, lower inference latency, and structural simplicity, the scratch-built **Advanced CNN (150 Epochs)** remains an incredibly efficient and powerful alternative.

### *Developed for the **Artificial Intelligence and Data Science** course at the **University of Évora**, June 2026.*