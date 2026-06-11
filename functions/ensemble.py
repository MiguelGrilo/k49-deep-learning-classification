import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


# ======================
# CLASSIFIER 5: ENSEMBLE
# ======================
def evaluate_ensemble(
    models_list,
    x_test, y_test, iter_num=1,
    description="Softmax Averaging Ensemble"
):
    """
    Evaluates an ensemble of trained Keras models using Softmax Averaging.

    Arguments:
    - models_list: List of trained tf.keras models.
    - x_test: Testing images.
    - y_test: True labels.
    - iter_num: Iteration number for terminal output.
    - description: Brief description of the ensemble strategy.
    """

    if description : print(f"# {description}")
    print(f"Iteration {iter_num}\nGenerating predictions for all models in the ensemble (This might take a few seconds)...")

    # Collect predictions from all models (Output shape: [num_models, num_samples, 49])
    all_predictions = np.array(
        [model.predict(x_test, verbose=0) for model in models_list]
    )

    # Average the probabilities across all models (Output shape: [num_samples, 49])
    avg_predictions = np.mean(all_predictions, axis=0)

    # Get the final class by picking the highest average probability
    final_pred = np.argmax(avg_predictions, axis=1)

    # Evaluation using Advanced Metrics
    acc = balanced_accuracy_score(y_test, final_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, final_pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, avg_predictions