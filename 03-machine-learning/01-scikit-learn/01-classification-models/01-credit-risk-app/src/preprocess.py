from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
	df = df.copy()

	# Encode category column
	le = LabelEncoder()
	df["marital_status"] = le.fit_transform(df["marital_status"])

	x = df.drop("target", axis=1)
	y = df["target"]

	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)

	return x_scaled, y, scaler
