from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_models():
	return{
		"linear_regression": LinearRegression(),
		"decision_tree": DecisionTreeRegressor(),
		"random_forest": RandomForestRegressor
	}
