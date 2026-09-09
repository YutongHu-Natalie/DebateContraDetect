import numpy as np
from sklearn.metrics import recall_score, precision_score
from typing import List, Dict, Union
from tabulate import tabulate


class ClassificationMetricsCalculator:
    """
    A class to calculate and display comprehensive classification metrics
    including per-label and macro-averaged precision and recall.
    """

    def __init__(self, y_true: List[int], predictions: List[List[int]]):
        """
        Initialize the calculator with ground truth and predictions.

        Args:
            y_true: List of true labels
            predictions: List of prediction arrays
        """
        self.y_true = np.array(y_true)
        self.predictions = [np.array(pred) for pred in predictions]
        self.unique_labels = sorted(set(self.y_true) |
                                    set(label for pred in self.predictions
                                        for label in pred))

    def calculate_metrics(self) -> List[Dict[str, Union[float, Dict[str, float]]]]:
        """
        Calculate precision and recall for each prediction array.

        Returns:
            List of dictionaries containing metrics for each prediction
        """
        all_metrics = []

        for i, y_pred in enumerate(self.predictions):
            metrics = {
                'prediction_id': i + 1,
                'per_label_precision': {},
                'per_label_recall': {},
            }

            # Calculate per-label metrics
            for label in self.unique_labels:
                # Precision for current label
                precision = precision_score(
                    self.y_true,
                    y_pred,
                    labels=[label],
                    average='macro',
                    zero_division=0
                )
                metrics['per_label_precision'][label] = precision

                # Recall for current label
                recall = recall_score(
                    self.y_true,
                    y_pred,
                    labels=[label],
                    average='macro',
                    zero_division=0
                )
                metrics['per_label_recall'][label] = recall

            # Calculate macro averages
            metrics['macro_precision'] = precision_score(
                self.y_true,
                y_pred,
                average='macro',
                zero_division=0
            )
            metrics['macro_recall'] = recall_score(
                self.y_true,
                y_pred,
                average='macro',
                zero_division=0
            )

            all_metrics.append(metrics)

        return all_metrics

    def display_metrics(self, metrics: List[Dict[str, Union[float, Dict[str, float]]]]):
        """
        Display the metrics in a formatted table.

        Args:
            metrics: List of metric dictionaries from calculate_metrics()
        """
        for prediction_metrics in metrics:
            pred_id = prediction_metrics['prediction_id']
            print(f"\nMetrics for Prediction {pred_id}")
            print("=" * 40)

            # Per-label metrics table
            headers = ['Label', 'Precision', 'Recall']
            rows = []
            for label in self.unique_labels:
                rows.append([
                    label,
                    f"{prediction_metrics['per_label_precision'][label]:.3f}",
                    f"{prediction_metrics['per_label_recall'][label]:.3f}"
                ])
            print(tabulate(rows, headers=headers, tablefmt='grid'))

            # Macro averages
            print("\nMacro Averages:")
            print(f"Macro Precision: {prediction_metrics['macro_precision']:.3f}")
            print(f"Macro Recall: {prediction_metrics['macro_recall']:.3f}")


# Example usage with your data
if __name__ == "__main__":
    y_true = [0,0,0,0,0,2,4,0,0,4,0,0,0,4,2,0]
    y_pred_list = [
        [2, 0, 2,0,0,2,4,2,2,2,2,2,4,4,2,0],
        [2, 0, 2,0,0,2,4,4,0,4,0,0,4,4,2,0],
        [0, 0, 0, 0, 0,2,4,0,0,1,0,0,4,4,2,0],
        [1,0,1,0,0,1,4,3,3,1,1,0,4,4,2,0]
    ]

    calculator = ClassificationMetricsCalculator(y_true, y_pred_list)
    metrics = calculator.calculate_metrics()
    calculator.display_metrics(metrics)