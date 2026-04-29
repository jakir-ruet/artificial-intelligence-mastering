from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(x, y):
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.2, random_state=42
	)

	model = RandomForestClassifier(n_estimators=100)
	model.fit(x_train, y_train)

	print("Model trained successfully")

	return model, x_test, y_test
