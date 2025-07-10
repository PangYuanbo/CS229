import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    MSE=1
    BestTau=None
    for i in tau_values:
        LWR = LocallyWeightedLinearRegression(i)
        LWR.fit(x_train, y_train)
        y_pred = LWR.predict(x_valid)
        y_pred=y_pred.flatten()
        y_valid=y_valid.flatten()
        mse=np.mean((y_pred - y_valid)**2)
        print("Tau:"+str(i)+", MSE:"+str(mse))
        plt.figure(figsize=(8, 5))
        plt.plot(x_valid[:, 1], y_valid, 'bx', label="valid")  # blue 'x' for training data
        plt.plot(x_valid[:, 1], y_pred, 'ro', label="predictions")  # red 'o' for predictions
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"LWR with tau = {i}")
        plt.legend()
        plt.show()
        if mse<=MSE :
            MSE=mse
            BestTau=i

    LWR = LocallyWeightedLinearRegression(BestTau)
    LWR.fit(x_train, y_train)
    y_pred = LWR.predict(x_test)
    mse=np.mean((y_pred - y_test)**2)
    plt.figure(figsize=(8, 5))
    plt.plot(x_test[:, 1], y_test, 'bx', label="test")
    plt.plot(x_test[:, 1], y_pred, 'ro', label="predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"LWR with BestTau = {BestTau}")
    plt.legend()
    plt.show()
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
