from typing import List
import matplotlib.pyplot as plt

def plot_training_metrics(epochs: List[int], test_accuracy: List[float], f1_scores: List[float], mcc_scores: List[float]) -> None:
    """
    Plot the relationship between test accuracy, F1 Score, and Matthews Correlation Coefficient over training epochs.

    :param epochs: A list of training epochs.
    :param test_accuracy: A list of test accuracy values.
    :param f1_scores: A list of F1 Score values.
    :param mcc_scores: A list of Matthews Correlation Coefficient values.
    """
    plt.figure(figsize=(10, 6))

    # Plot test accuracy
    plt.plot(epochs, test_accuracy, label='Test Accuracy', color='blue', marker='o')

    # Plot F1 Score
    plt.plot(epochs, f1_scores, label='F1 Score', color='green', marker='s')

    # Plot Matthews Correlation Coefficient
    plt.plot(epochs, mcc_scores, label='Matthews Correlation Coefficient', color='red', marker='^')

    # Add legend
    plt.legend()

    # Chart title
    plt.title('Training Metrics Over Epochs')

    # Axis labels
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')

    # Show grid
    plt.grid(True)

    # Display the chart
    plt.show()


