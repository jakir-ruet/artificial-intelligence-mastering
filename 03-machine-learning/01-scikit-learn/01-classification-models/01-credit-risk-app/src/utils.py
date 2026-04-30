from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, x_test, y_test):
	pred = model.predict(x_test)

	print("Accuracy: ", accuracy_score(y_test, pred))
	print("\nReport: \n", classification_report(y_test, pred))
