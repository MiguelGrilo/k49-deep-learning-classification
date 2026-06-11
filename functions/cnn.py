import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


# =================
# CLASSIFIER 3: CNN
# =================
def advanced_cnn(
    x_train,
    y_train,
    x_test,
    y_test,
    iter_num=1,
    dense_neurons=128,
    num_conv_blocks=2,
    use_data_augmentation=False,
    aug_rotation=0.05,
    use_batch_norm=False,
    use_l2_reg=False,
    use_lr_scheduler=False,
    dropout_rate=0.3,
    learning_rate=0.001,
    epochs=30,
    batch_size=256,
    early_stop_patience=3,
    description="Baseline CNN parameters.",
):
    # Lazy Imports
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    """Trains and evaluates a highly configurable CNN for the K49 dataset."""

    if description:
        print(f"# {description}")

    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # Dynamically build the Sequential model
    model_cnn = models.Sequential()
    model_cnn.add(layers.Input(shape=(28, 28, 1)))

    # Data Augmentation
    if use_data_augmentation:
        data_augmentation = models.Sequential(
            [
                layers.RandomRotation(factor=aug_rotation, fill_mode="constant"),
                layers.RandomTranslation(
                    height_factor=0.1, width_factor=0.1, fill_mode="constant"
                ),
            ],
            name="data_augmentation",
        )
        model_cnn.add(data_augmentation)

    # Regularization logic
    reg = regularizers.l2(1e-4) if use_l2_reg else None

    # Block 1
    model_cnn.add(layers.Conv2D(32, (3, 3), kernel_regularizer=reg))
    if use_batch_norm:
        model_cnn.add(layers.BatchNormalization())
    model_cnn.add(layers.Activation("relu"))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    # Block 2
    model_cnn.add(layers.Conv2D(64, (3, 3), kernel_regularizer=reg))
    if use_batch_norm:
        model_cnn.add(layers.BatchNormalization())
    model_cnn.add(layers.Activation("relu"))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    # Block 3
    if num_conv_blocks >= 3:
        model_cnn.add(layers.Conv2D(128, (3, 3), kernel_regularizer=reg))
        if use_batch_norm:
            model_cnn.add(layers.BatchNormalization())
        model_cnn.add(layers.Activation("relu"))

    # Classification Head
    model_cnn.add(layers.Flatten())

    # Dense Layer
    model_cnn.add(layers.Dense(dense_neurons, kernel_regularizer=reg))
    if use_batch_norm:
        model_cnn.add(layers.BatchNormalization())
    model_cnn.add(layers.Activation("relu"))

    model_cnn.add(layers.Dropout(dropout_rate))
    model_cnn.add(layers.Dense(49, activation="softmax"))

    # Compile
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_cnn.compile(optimizer=optimizer_adam, loss="sparse_categorical_crossentropy")

    # Callbacks
    callbacks_list = [
        EarlyStopping(
            monitor="val_loss", patience=early_stop_patience, restore_best_weights=True
        )
    ]

    if use_lr_scheduler:
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        )
        callbacks_list.append(reduce_lr)

    print(f"Iteration {iter_num}\nTraining CNN...")
    model_cnn.fit(
        x_train_cnn,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1,
    )

    prob = model_cnn.predict(x_test_cnn, verbose=0)
    pred = np.argmax(prob, axis=1)
    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_cnn


# ==========================
# CLASSIFIER 8: RESIDUAL CNN
# ==========================
def residual_cnn(
    x_train,
    y_train,
    x_test,
    y_test,
    iter_num=1,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    description="Baseline Residual CNN with 1 Skip Connection.",
):
    # Lazy Imports
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    """
    Builds, trains and evaluates a CNN with a Residual Block using the Functional API.
    """

    if description:
        print(f"# {description}")

    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # --- MODEL ARCHITECTURE (Functional API) ---
    inputs = layers.Input(shape=(28, 28, 1))

    # Initial Convolution
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # --- RESIDUAL BLOCK START ---
    shortcut = x  # Save the input of the block

    # Main path
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)  # No activation yet
    x = layers.BatchNormalization()(x)
    # Add the shortcut to the main path (The Skip Connection)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)  # Apply activation after addition
    # --- RESIDUAL BLOCK END ---

    # Classification Head
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(49, activation="softmax")(x)

    model_res = models.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_res.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    print(
        f"Iteration {iter_num}\nTraining Residual CNN (Epochs: {epochs}, Batch: {batch_size})..."
    )
    model_res.fit(
        x_train_cnn,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    prob = model_res.predict(x_test_cnn, verbose=0)
    pred = np.argmax(prob, axis=1)

    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_res


# =========================================
# CLASSIFIER 9: CNN WITH MIXUP AUGMENTATION
# =========================================
def mixup_cnn(
    x_train, y_train, x_test, y_test, iter_num=1,
    epochs=40,
    batch_size=256,
    learning_rate=0.001,
    description="CNN with MixUp Data Augmentation.",
):
    # Lazy Imports
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    """
    Builds, trains and evaluates a CNN using MixUp Data Augmentation.
    """

    if description : print(f"# {description}")

    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # Manual Validation Split (80% Train, 20% Val)
    split_idx = int(len(x_train_cnn) * 0.8)
    x_train_split, x_val_split = x_train_cnn[:split_idx], x_train_cnn[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

    # Convert all labels to One-Hot Encoding
    y_train_split_oh = tf.keras.utils.to_categorical(y_train_split, num_classes=49)
    y_val_split_oh = tf.keras.utils.to_categorical(y_val_split, num_classes=49)

    # Prepare MixUp Dataset function
    def prepare_mixup_dataset(x, y, batch_sz=256):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_sz)

        def mixup(images, labels):
            weight = tf.random.uniform([], minval=0.0, maxval=1.0)
            indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
            shuffled_images = tf.gather(images, indices)
            shuffled_labels = tf.gather(labels, indices)

            labels = tf.cast(labels, tf.float32)
            shuffled_labels = tf.cast(shuffled_labels, tf.float32)

            blended_images = weight * images + (1.0 - weight) * shuffled_images
            blended_labels = weight * labels + (1.0 - weight) * shuffled_labels
            return blended_images, blended_labels

        return (dataset
                .map(mixup, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    # Generate the MixUp training dataset
    train_ds = prepare_mixup_dataset(x_train_split, y_train_split_oh, batch_size)

    # Build CNN Architecture
    reg = regularizers.l2(1e-4)
    model_mixup = models.Sequential(
        [
            layers.InputLayer(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(49, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_mixup.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the Model
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    print(f"Iteration {iter_num}\nTraining CNN with MixUp (Epochs: {epochs}, Batch: {batch_size})...")
    model_mixup.fit(
        train_ds,
        epochs=epochs,
        validation_data=(x_val_split, y_val_split_oh),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    prob = model_mixup.predict(x_test_cnn, verbose=0)
    pred = np.argmax(prob, axis=1)

    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_mixup