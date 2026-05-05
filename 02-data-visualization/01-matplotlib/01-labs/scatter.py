import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [5,7,6,8,7]

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()
plt.savefig("output-scatter-plot.png")
