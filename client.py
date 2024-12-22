import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        # Return model weights (coefficients and intercept)
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        # Set model weights (coefficients and intercept)
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        # Set model parameters and train
        self.set_parameters(parameters)
        self.model.fit(self.x_train, self.y_train)
        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Set model parameters and evaluate
        self.set_parameters(parameters)
        loss = 1.0 - self.model.score(
            self.x_test, self.y_test
        )  # Accuracy as 1 - accuracy = loss
        return loss, len(self.x_test), {}


def start_client():
    # Load dataset
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Create model and initialize coefficients
    model = LogisticRegression(max_iter=100)
    model.fit([[0, 0, 0, 0]], [0])  # Dummy training to initialize weights

    # Start Flower client
    client = FederatedClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)


if __name__ == "__main__":
    start_client()
