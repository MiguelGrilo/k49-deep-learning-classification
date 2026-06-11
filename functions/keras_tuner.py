import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from functions.paths import KERAS_TUNER_DIR


# =========================
# CLASSIFIER 7: KERAS TUNER
# =========================
def keras_tuner(
    x_train, y_train, x_test, y_test, iter_num=1,
    max_trials=5,
    epochs=10,
    description="Hyperparameter Optimization with KerasTuner",
):
    # Lazy Imports
    import keras_tuner as kt
    import tensorflow as tf
    from tensorflow.keras import layers, models

    """
    Runs KerasTuner RandomSearch to find the optimal CNN architecture for K49.
    """

    if description : print(f"# {description}")

    x_train_cnn = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test_cnn = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # Define the Builder Function inside the wrapper
    def build_tunable_model(hp):
        model = models.Sequential()
        model.add(layers.InputLayer(shape=(28, 28, 1)))

        # Tune the number of filters in the first Conv2D layer (min: 32, max: 128, step: 32)
        hp_filters = hp.Int("filters_1", min_value=32, max_value=128, step=32)
        model.add(layers.Conv2D(hp_filters, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        # Tune the Dense layer units
        hp_units = hp.Int("dense_units", min_value=128, max_value=512, step=128)
        model.add(layers.Dense(hp_units, activation="relu"))

        # Tune the Dropout rate
        hp_dropout = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
        model.add(layers.Dropout(hp_dropout))

        model.add(layers.Dense(49, activation="softmax"))

        # Tune the Learning Rate
        hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Initialize the Tuner
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective="val_accuracy",
        max_trials=max_trials,
        directory=KERAS_TUNER_DIR,
        project_name=f"k49_optimization_iter_{iter_num}",
    )

    # Execute the Search
    print(f"\nStarting Hyperparameter Search (Max Trials: {max_trials}, Epochs per trial: {epochs})...")
    tuner.search(x_train_cnn, y_train, epochs=epochs, validation_split=0.2, verbose=1)

    # Extract the absolute Best Model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    print("\n# ----------------------------------------------------")
    print("# OPTIMAL HYPERPARAMETERS FOUND:")
    print(f"# - Conv2D Filters: {best_hp.get('filters_1')}")
    print(f"# - Dense Units:    {best_hp.get('dense_units')}")
    print(f"# - Dropout Rate:   {best_hp.get('dropout')}")
    print(f"# - Learning Rate:  {best_hp.get('learning_rate')}")
    print("# ----------------------------------------------------\n")

    print("Evaluating the best model on Test Data...")
    prob = best_model.predict(x_test_cnn, verbose=0)
    pred = np.argmax(prob, axis=1)

    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, best_model