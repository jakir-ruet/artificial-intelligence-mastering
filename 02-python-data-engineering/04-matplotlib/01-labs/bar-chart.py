import matplotlib.pyplot as plt

categories = ['IT', 'HR', 'Finance']
values = [50, 30, 20]

plt.bar(categories, values)
plt.title("Bar Chart")
plt.show()
plt.savefig("output-bar-plot.png")
