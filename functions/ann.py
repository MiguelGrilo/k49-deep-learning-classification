import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


# ========================
# CLASSIFIER 1: SIMPLE ANN
# ========================
def simple_ann(
    x_train, y_train, x_test, y_test, iter_num=1,
    hidden_neurons=128,
    epochs=5,
    batch_size=128,
    dropout_rate=0.2,
    activation_func="relu",
    optimizer_name="adam",
    loss_name="sparse_categorical_crossentropy",
    description="Baseline parameters.",
):
    # Lazy Imports
    from tensorflow.keras import layers, models

    """Trains and evaluates a Simple ANN for the K49 dataset."""

    if description : print(f"# {description}")

    model_ann = models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(hidden_neurons, activation=activation_func),
            layers.Dropout(dropout_rate),
            layers.Dense(49, activation="softmax"),
        ]
    )

    model_ann.compile(optimizer=optimizer_name, loss=loss_name)
    print(f"Iteration {iter_num}\nTraining Simple ANN (Neurons: {hidden_neurons}, Activation: {activation_func}, Optimizer: {optimizer_name}, Epochs: {epochs}, Batch: {batch_size}, Dropout: {dropout_rate})...")
    model_ann.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluation using Advanced Metrics
    prob = model_ann.predict(x_test, verbose=0)
    pred = np.argmax(prob, axis=1)
    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_ann


# ======================
# CLASSIFIER 2: DEEP ANN
# ======================
def deep_ann(
    x_train, y_train, x_test, y_test, iter_num=1,
    layer1_neurons=300,
    layer2_neurons=100,
    epochs=50,
    batch_size=256,
    dropout_rate=0.3,
    early_stop_patience=5,
    activation_func="relu",
    optimizer_name="adam",
    loss_name="sparse_categorical_crossentropy",
    description="Baseline Deep ANN parameters.",
):
    # Lazy Imports
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping

    """Trains and evaluates a Deep ANN for the K49 dataset."""

    if description : print(f"# {description}")

    model_deep = models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(layer1_neurons, activation=activation_func),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(layer2_neurons, activation=activation_func),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(49, activation="softmax"),
        ]
    )

    model_deep.compile(optimizer=optimizer_name, loss=loss_name)
    early_stop_ann = EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True)
    print(f"Iteration {iter_num}\nTraining Deep ANN (L1: {layer1_neurons}, L2: {layer2_neurons}, Epochs: {epochs}, Batch: {batch_size}, Dropout: {dropout_rate})...")
    model_deep.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop_ann],
        verbose=1,
    )

    # Evaluation using Advanced Metrics
    prob = model_deep.predict(x_test, verbose=0)
    pred = np.argmax(prob, axis=1)
    acc = balanced_accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_deep


# =========================
# CLASSIFIER 4: PYTORCH ANN
# =========================
def pytorch_ann(
    x_train, y_train, x_test, y_test, iter_num=1,
    hidden_neurons=128,
    epochs=5,
    batch_size=256,
    optimizer_name="sgd",
    learning_rate=0.1,
    description="Baseline PyTorch parameters.",
):
    # Lazy Imports
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    """Trains and evaluates a Simple ANN using PyTorch for the K49 dataset."""

    if description : print(f"# {description}")

    # Define PyTorch Dataset locally to encapsulate logic
    class K49Data(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X.reshape(-1, 28 * 28).astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.int64))
            self.len = self.X.shape[0]
        def __getitem__(self, index):
            return self.X[index], self.y[index]
        def __len__(self):
            return self.len

    # Create DataLoaders
    train_data_pt = K49Data(x_train, y_train)
    train_dataloader = DataLoader(
        dataset=train_data_pt, batch_size=batch_size, shuffle=True
    )
    test_data_pt = K49Data(x_test, y_test)
    test_dataloader = DataLoader(
        dataset=test_data_pt, batch_size=batch_size, shuffle=False
    )

    # Define the PyTorch Model locally
    input_dim = 28 * 28
    output_dim = 49

    class NeuralNetworkPT(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(NeuralNetworkPT, self).__init__()
            self.layer_1 = nn.Linear(input_dim, hidden_dim)
            nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
            self.layer_2 = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = torch.nn.functional.relu(self.layer_1(x))
            x = self.layer_2(x)
            return x

    # Initialization and Device placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pt = NeuralNetworkPT(input_dim, hidden_neurons, output_dim).to(device)

    # Optimizer Selection
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_pt.parameters(), lr=learning_rate) if optimizer_name.lower() == "adam" \
        else torch.optim.SGD(model_pt.parameters(), lr=learning_rate)

    print(f"Iteration {iter_num}\nTraining PyTorch NN on {device} (Neurons: {hidden_neurons}, Opt: {optimizer_name.upper()}, Epochs: {epochs})...")

    for epoch in range(epochs):
        print(f" -> Epoch {epoch + 1:02d}/{epochs}... ", end="", flush=True)

        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model_pt(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        print("Done!", flush=True)

    print("\nGenerating predictions on Test Set...", flush=True)
    # Training Loop
    print(f"Iteration {iter_num}\nTraining PyTorch NN on {device} (Neurons: {hidden_neurons}, Opt: {optimizer_name.upper()}, LR: {learning_rate}, Epochs: {epochs})...")
    for epoch in range(epochs):
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model_pt(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation Loop
    all_preds = []
    all_targets = []
    all_probs = []

    softmax_fn = nn.Softmax(dim=1)
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            outputs = model_pt(X_batch)
            # Extract probabilities
            probs_batch = softmax_fn(outputs)
            all_probs.extend(probs_batch.cpu().numpy())
            # Extract classes
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    prob = np.array(all_probs)

    # Evaluation using Advanced Metrics
    acc = balanced_accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, prob, model_pt