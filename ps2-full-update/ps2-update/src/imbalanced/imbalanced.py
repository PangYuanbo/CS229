import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    clf = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf.fit(x_train, y_train)
    x_validation, y_validation = util.load_dataset(validation_path, add_intercept=True)
    y_pred = clf.predict(x_validation)
    np.savetxt(output_path_naive, y_pred, delimiter=',')
    TP= np.sum((y_validation == 1) & (y_pred >= 0.5))
    TN= np.sum((y_validation == 0) & (y_pred < 0.5))
    FP= np.sum((y_validation == 0) & (y_pred >= 0.5))
    FN= np.sum((y_validation == 1) & (y_pred < 0.5))
    accuracy = (TP + TN) / len(y_validation)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    average_accuracy = 1/2 * (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    util.plot(x_validation, y_validation, clf.theta, output_path_naive.replace('.txt', '.png'))
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_upsampled = []
    y_upsampled = []
    for i in range(len(x_train)):
        if y_train[i] == 1:
            x_upsampled.append(x_train[i])
            y_upsampled.append(y_train[i])
        else:
            for _ in range(int(1 / kappa)):
                x_upsampled.append(x_train[i])
                y_upsampled.append(y_train[i])
    x_upsampled = np.array(x_upsampled)
    y_upsampled = np.array(y_upsampled)
    clf_upsampled = LogisticRegression()
    clf_upsampled.fit(x_upsampled, y_upsampled)
    y_pred_upsampled = clf_upsampled.predict(x_validation)
    np.savetxt(output_path_upsampling, y_pred_upsampled, delimiter=',')
    TP = np.sum((y_validation == 1) & (y_pred_upsampled >= 0.5))
    TN = np.sum((y_validation == 0) & (y_pred_upsampled < 0.5))
    FP = np.sum((y_validation == 0) & (y_pred_upsampled >= 0.5))
    FN = np.sum((y_validation == 1) & (y_pred_upsampled < 0.5))
    accuracy = (TP + TN) / len(y_validation)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    average_accuracy = 1 / 2 * (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    util.plot(x_validation, y_validation, clf_upsampled.theta, output_path_upsampling.replace('.txt', '.png'))

    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
