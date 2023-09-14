from reservoir import MemristorSim
import numpy as np
import networkx as nx
from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score


# Hardcoded in, but can easily be added as a model parameter.
simParams = {
            'MU': 3.46e-5,
            'KAPPA': 0.038,
            'R_ON': 13e3,
            'R_OFF': 13e3 / 1e-2,
            'z_0': 0.05,
            'dist': 'normal',
            'var': 0,
            'NOISE': 0.0,
            'THETA': 10
        }


class MemristorModel:
    def __init__(self, N : int, v_min : float, v_max : float, n_outputs : int) -> None:
        """_summary_

        Args:
            N (int): Number of nodes in memristive network
            v_min (float): Minimum voltage
            v_max (float): Maximum voltage
            n_outputs (int): Number of outputs
        """
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.n_outputs = n_outputs

        self.W = self._create_adjacency_matrix(self.N)
        self.readout = MultiOutputRegressor(LinearRegression())
        self.voltage_scaler = MinMaxScaler(feature_range=(self.v_min, self.v_max))
        self.readout_scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False)


    def _create_adjacency_matrix(self, shape : int = 10) -> np.ndarray:
        """Generates a shape x shape adjacency matrix. Each edge corresponds to a memristive element.

        Args:
            shape (int, optional): Size of adjacency matrix (i.e. the number of vertices in network graph). Defaults to 10.

        Returns:
            np.ndarray: Adjacency matrix.
        """
        while True:
            try:
                W = np.random.randint(0,2, (shape,shape))
                W ^= W.T
                if nx.is_connected(nx.from_numpy_array(W)) == True:
                    print("Connected network found")
                    return W
            except:
                "Disconnected network, trying again..."
    

    def _apply_voltage(self, inputs : np.ndarray, verbose : bool = True) -> np.ndarray:
        """Applies the voltage input sequence to the memristor network. A new network is instanstiated each time.

        Args:
            inputs (np.ndarray): Input sequence
            verbose (bool, optional): Set to True produces tqdm progress bars. Defaults to True.

        Returns:
            np.ndarray: Output of network, flattened to a single output stream.
        """
        outputs = np.zeros((*inputs.shape, self.n_outputs))
        for i in tqdm(range(inputs.shape[0]), disable=~verbose):
            Vext = inputs[i, :].reshape((-1,1))
            sim = MemristorSim(self.W, simParams)
            output, _ = sim.applyVoltage(Vext, outputs=self.n_outputs)
            outputs[i,...] = output

        outputs_sel = outputs[:, 9::10, :]
        flattened_outputs = outputs_sel.reshape((outputs_sel.shape[0], outputs_sel.shape[1]*outputs_sel.shape[2]))

        return flattened_outputs
    

    def fit(self, X : np.ndarray, y : np.ndarray, verbose : bool = True) -> None:
        """_summary_

        Args:
            X (np.ndarray): Training samples
            y (np.ndarray): Training targets
            verbose (bool): Set to True produces tqdm progress bars. Defaults to True
        """
        self.voltage_scaler.fit(X)
        X_scaled = self.voltage_scaler.transform(X).repeat(10, axis=1)
        flattened_outputs = self._apply_voltage(X_scaled, verbose)

        self.readout_scaler.fit(flattened_outputs)
        scaled_outputs = self.readout_scaler.transform(flattened_outputs)

        self.encoder.fit(y.reshape(-1, 1))
        y_onehot = self.encoder.transform(y.reshape(-1, 1))

        self.readout.fit(scaled_outputs, y_onehot)


    def predict(self, X : np.ndarray, verbose=True) -> np.ndarray:
        """_summary_

        Args:
            X (np.ndarray): Testing data
            verbose (bool, optional): Set to True produces tqdm progress bars. Defaults to True.

        Returns:
            (np.ndarray): Predictions
        """
        X_scaled = self.voltage_scaler.transform(X).repeat(10, axis=1)
        flattened_outputs = self._apply_voltage(X_scaled, verbose)

        scaled_outputs = self.readout_scaler.transform(flattened_outputs)
        y_pred = self.readout.predict(scaled_outputs)

        return y_pred
    

def get_digits_data(validation_size : float = 0.25, test_size : float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the digits data and split into training, testing, validation sets

    Args:
        validation_size (float, optional): Proportion of validation data. Defaults to 0.25.
        test_size (float, optional): Proportion of testing data. Defaults to 0.25.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_val, y_val, X_test, y_test
    """
    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, y_train, X_val, y_val, X_test, y_test


def score(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """_summary_

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted one hot encoded labels

    Returns:
        float: Accuracy
    """
    return accuracy_score(y_true, y_pred.argmax(axis=1))


