from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

y_true = [0,1,0,1,1]
y_pred = [0,1,0,0,1]

cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap='Blues')
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center')

plt.title("Confusion Matrix")
plt.savefig("confusion-matrix.png")
