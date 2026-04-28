def validate(df):
	required_cols = ['id', 'name', 'age', 'department', 'salary', 'date']

	for col in required_cols:
		if col not in df.columns:
			raise ValueError(f"The missing column: {col}")
