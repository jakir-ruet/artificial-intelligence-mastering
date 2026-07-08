from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from .config import Config

def load_data():
	X, y = make_regression(
		n_samples=100,
		n_features=4,
		noise=0.1,
		random_state=Config.RANDOM_STATE
	)
	return train_test_split(
		X, y,
		test_size=Config.TEST_SIZE,
		random_state=Config.RANDOM_STATE
	)
