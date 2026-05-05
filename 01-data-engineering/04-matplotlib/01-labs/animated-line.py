import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = []
y = []

fig, ax = plt.subplots()

def update(frame):
    x.append(frame)
    y.append(frame * 2)

    ax.clear()
    ax.plot(x, y)
    ax.set_title("Animated Plot")

ani = FuncAnimation(fig, update, frames=range(10), interval=500)

# ✅ Save instead of show
ani.save("output-animated-line.gif")
