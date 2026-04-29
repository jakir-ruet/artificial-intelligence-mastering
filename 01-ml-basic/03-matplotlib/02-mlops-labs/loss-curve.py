import matplotlib.pyplot as plt

epochs = [1,2,3,4,5]
train_loss = [0.9, 0.7, 0.5, 0.4, 0.3]
val_loss = [1.0, 0.8, 0.6, 0.55, 0.5]

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig("loss-curve.png")

# Used to detect:
# 1. Overfitting
# 2. Underfitting
