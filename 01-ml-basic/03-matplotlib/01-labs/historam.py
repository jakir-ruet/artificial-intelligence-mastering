import matplotlib.pyplot as plt

data = [22, 25, 29, 30, 35, 40, 40, 42]

plt.hist(data, bins=5)
plt.title("Histogram")
plt.show()
plt.savefig("output-histogram-plot.png")
