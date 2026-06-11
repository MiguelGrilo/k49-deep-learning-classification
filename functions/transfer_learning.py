import os
import cv2
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


# ========================================
# CLASSIFIER 6: TRANSFER LEARNING (RESNET)
# ========================================
def tl_resnet(
    x_train, y_train, x_test, y_test, iter_num=1,
    epochs=10,
    batch_size=256,
    learning_rate=0.001,
    fine_tune=False,
    fine_tune_epochs=10,
    fine_tune_lr=1e-5,
    img_size=32,
    use_augmentation=False,
    pooling_type="avg",
    description="Baseline Transfer Learning with ResNet50V2.",
):
    # Lazy Imports
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    """
    Builds, trains and evaluates a Transfer Learning model.
    Supports Fine-Tuning, dynamic resizing, data augmentation and pooling strategies.
    """

    if description : print(f"# {description}")

    x_train_tl = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test_tl = x_test.reshape(-1, 28, 28, 1).astype("float32")

    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    # Adapt Data (Resize and duplicate channels to simulate RGB)
    x = layers.Resizing(img_size, img_size)(x)
    x = layers.Concatenate(axis=-1)([x, x, x])

    # Load Pre-trained Model
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # PHASE 1: Freeze the base model
    base_model.trainable = False
    x = base_model(x, training=False)

    # Pooling Strategy
    x = layers.GlobalMaxPooling2D()(x) if pooling_type == "max" \
        else layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(49, activation="softmax")(x)

    model_tl = models.Model(inputs, outputs)

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Compile Phase 1 (Warmup)
    model_tl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nIteration {iter_num}\n--- PHASE 1: Training Head (Epochs: {epochs}, LR: {learning_rate}) ---")

    if use_augmentation:
        print("   [+] Using CPU-based Data Augmentation (GPU Safe Mode)...")

        # Define the augmentation model safely outside the main graph
        data_augmentation = models.Sequential(
            [
                layers.RandomRotation(0.05),
                layers.RandomTranslation(0.1, 0.1),
            ]
        )

        # Manually create the 20% validation split
        split_idx = int(len(x_train_tl) * 0.8)
        x_train_split, y_train_split = x_train_tl[:split_idx], y_train[:split_idx]
        x_val_split, y_val_split = x_train_tl[split_idx:], y_train[split_idx:]

        # Create CPU pipelines
        train_ds = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
        train_ds = train_ds.shuffle(1024).batch(batch_size)

        # Apply augmentation on the fly (training only)
        train_ds = train_ds.map(
            lambda img, label: (data_augmentation(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        val_ds = (tf.data.Dataset
            .from_tensor_slices((x_val_split, y_val_split))
            .batch(batch_size)
        )

        model_tl.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        # If fine-tuning is required, we use the same datasets
        if fine_tune:
            print(f"\n--- PHASE 2: Fine-Tuning Base Model (Epochs: {fine_tune_epochs}, LR: {fine_tune_lr}) ---")
            base_model.trainable = True
            model_tl.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            early_stop_ft = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            model_tl.fit(
                train_ds,
                epochs=fine_tune_epochs,
                validation_data=val_ds,
                callbacks=[early_stop_ft, reduce_lr],
                verbose=1,
            )

    else:
        # Standard training without augmentation
        model_tl.fit(
            x_train_tl, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        if fine_tune:
            print(f"\n--- PHASE 2: Fine-Tuning Base Model (Epochs: {fine_tune_epochs}, LR: {fine_tune_lr}) ---")
            base_model.trainable = True
            model_tl.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            early_stop_ft = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            model_tl.fit(
                x_train_tl, y_train,
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stop_ft, reduce_lr],
                verbose=1,
            )

    print("\nEvaluating model on Test Set...")
    prob = model_tl.predict(x_test_tl, verbose=0)
    pred = np.argmax(prob, axis=1)

    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_tl


# =====================
# CLASSIFIER 10: YOLOv8
# =====================
def tl_yolo(
    x_train, y_train, x_test, y_test, iter_num=1,
    epochs=10,
    batch_size=256,
    img_size=32,
    description="YOLOv8-cls Transfer Learning",
):
    # Lazy Import
    from ultralytics import YOLO

    """
    Exports numpy arrays to a YOLO-compatible folder structure and trains YOLOv8-cls.
    """

    if description : print(f"# {description}")

    # YOLO requires images saved on disk in a specific folder structure:
    # dataset/yolo_format/train/class_0/img1.png
    base_dir = "dataset/yolo_format"

    # Export Data to Disk (Only if it hasn't been done yet)
    if not os.path.exists(base_dir):
        print(f"\n[INFO] Converting Numpy arrays to YOLO folder structure in '{base_dir}'...")
        print("[INFO] This will take a few minutes, but only happens ONCE.")

        for split_name, x_data, y_data in [
            ("train", x_train, y_train),
            ("val", x_test, y_test),
        ]:
            for i, (img, label) in enumerate(zip(x_data, y_data)):
                class_dir = os.path.join(base_dir, split_name, str(label))
                os.makedirs(class_dir, exist_ok=True)

                # Convert back to 0-255 scale for saving as image (CORRECT!)
                img_to_save = (img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(class_dir, f"img_{i}.png"), img_to_save)

        print("[INFO] Dataset export complete!\n")
    else:
        print(f"\n[INFO] YOLO dataset structure already found at '{base_dir}'. Skipping export.\n")

    # Initialize YOLOv8 Classification Model (Nano version for speed)
    run_dir = os.path.join("runs", "classify", "YOLO_K49_Runs", f"Iteration_{iter_num}")
    best_weights_path = os.path.join(run_dir, "weights", "best.pt")
    last_weights_path = os.path.join(run_dir, "weights", "last.pt")

    # ==============================================================
    # Check if the model is ALREADY fully trained
    # ==============================================================
    if os.path.exists(best_weights_path):
        print(f"\n[INFO] Fully trained model found at '{best_weights_path}'!")
        print("[INFO] Skipping training and loading best weights directly...\n")
        model_yolo = YOLO(best_weights_path)

    # ==============================================================
    # Auto-Resume logic if it crashed mid-training
    # ==============================================================
    elif os.path.exists(last_weights_path):
        print(f"\n[INFO] Found checkpoint from a previous crash: '{last_weights_path}'")
        print("[INFO] Resuming training from the last saved epoch...\n")
        model_yolo = YOLO(last_weights_path)
        model_yolo.train(resume=True)

    # ==============================================================
    # Start from scratch if no weights exist
    # ==============================================================
    else:
        print(f"\n[INFO] No checkpoint found. Starting fresh training session.")
        print(f"Training YOLOv8-cls (Epochs: {epochs}, Batch: {batch_size}, Img Size: {img_size})...")

        model_yolo = YOLO("weights/yolov8n-cls.pt")
        model_yolo.train(
            data=os.path.abspath(base_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project="YOLO_K49_Runs",
            name=f"Iteration_{iter_num}",
            exist_ok=True,
            verbose=False,
            # amp=False,
            # workers=0,
        )

    # Evaluation
    print("\nGenerating predictions on Test Set...")
    search_path = os.path.join(os.path.abspath(base_dir), "val", "**", "*.png")
    val_results = model_yolo.predict(
        source=search_path, imgsz=img_size, verbose=False, stream=True
    )

    pred = []
    y_true_yolo = []
    probs_list = []

    for r in val_results:
        predicted_class_name = r.names[r.probs.top1]
        pred.append(int(predicted_class_name))

        folder_class = int(os.path.basename(os.path.dirname(r.path)))
        y_true_yolo.append(folder_class)

        probs_list.append(r.probs.data.cpu().numpy())

    prob = np.array(probs_list)
    acc = balanced_accuracy_score(y_true_yolo, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_yolo, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_yolo