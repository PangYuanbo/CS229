import numpy as np
import util


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set.
    # Use save_path argument to save various visualizations for your own reference.
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_train)
    # util.write_json(save_path, {'theta': clf.theta.tolist(), 'predictions': predictions.tolist()})
    # util.plot_points(x_train, y_train)
    # util.plot_contour(predictions)
    import matplotlib.pyplot as plt
    plt.scatter(x_train[:, 1], x_train[:, 2], c=predictions, cmap='coolwarm', alpha=0.5)
    plt.title('Logistic Regression Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Predicted Probability')
    plt.savefig(save_path.replace('.txt', '.png'))
    # Plot the decision boundary
    x1 = np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), 100)
    x2 = -(clf.theta[0] / clf.theta[2] + clf.theta[1] / clf.theta[2] * x1)
    plt.plot(x1, x2, color='red', linewidth=2, label='Decision Boundary')
    plt.legend()
    plt.savefig(save_path.replace('.txt', '_boundary.png'))






    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        self.lamda = 0.01
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        for iter in range(self.max_iter):
            h_X=1/(1+np.exp(-self.theta.T@x.T))
            J_X=-np.mean(y * np.log(h_X) + (1 - y) * np.log(1 - h_X))+self.lamda*np.sum(self.theta**2)/2
            # Gradient of the loss function
            grad= x.T @ (h_X - y) / y.shape[0]+self.lamda*self.theta
            # Update theta
            update=self.learning_rate*grad
            if np.linalg.norm(update)<self.eps:
                break
            self.theta -= update
            if self.verbose:
                loss= np.mean(-y * np.log(h_X+self.eps) - (1 - y) * np.log(1 - h_X+self.eps))
                print(f'Loss after {iter} iterations: {loss:.4f}')
                # print(np.linalg.norm(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = 1 / (1 + np.exp(-self.theta.T @ x.T))
        return predictions
        # *** END CODE HERE ***

if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a.txt')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')
