import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from fuzzytree._classes import FuzzyDecisionTreeRegressor
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=500, noise=0.1):
    """
    Generate synthetic regression data: y = sin(x) + noise
    """
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + noise * np.random.randn(n_samples)
    return X, y

def test_fuzzy_regression_tree():
    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the fuzzy regression tree
    regressor = FuzzyDecisionTreeRegressor(
        fuzziness=0.8,
        criterion='variance',  # Assuming 'variance' is implemented for regression
        max_depth=5,
        min_membership_split=0.02,
        min_membership_leaf=0.01,
        min_impurity_decrease=0.001
    )

    # Fit the model
    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print("Fuzzy Regression Tree Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    plt.scatter(X_test, y_test, color='blue', label='True Values')
    plt.scatter(X_test, y_pred, color='red', label='Predictions')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Fuzzy Regression Tree Performance")
    plt.show()

if __name__ == "__main__":
    test_fuzzy_regression_tree()
