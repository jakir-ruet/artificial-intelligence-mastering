import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = []
y = []

fig, ax = plt.subplots()

def update(frame):
    x.append(frame)
    y.append(frame * 2)

    ax.clear()
    ax.scatter(x, y, color='blue')
    ax.set_title("Animated Scatter Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

ani = FuncAnimation(fig, update, frames=range(10), interval=500)

# ✅ Save animation
ani.save("output-scatter-line.gif", writer="pillow")
