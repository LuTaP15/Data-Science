"""
Plotting multiple plots in one figure with gridspec.
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Styling
plt.style.use("seaborn-whitegrid")
#plt.rcParams["font.family"] = "Avenir"
plt.rcParams["font.size"] = 12

# Create Blank Figure
fig = plt.figure(figsize=(10, 10))

fig.suptitle("Panneled figure with multiple plots", fontsize=16)

# Create 2x2 Grid
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1], width_ratios=[2, 1])

# Create Three Axes Objects
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# Dummy Data
x = np.linspace(0, 2*np.pi, 1000)


# Plot Data
ax1.plot(x, np.sin(x))
ax2.plot(x, np.sin(x)**2)
ax3.plot(x, np.sin(x) + np.cos(3*x))

# Set axis labels
ax1.set_ylabel("y", size=20)
ax3.set_xlabel("x", size=20)
ax3.set_ylabel("y", size=20)

# Set titles
ax1.set_title("Plot 1")
ax2.set_title("Plot 2")
ax3.set_title("Plot 3")

plt.show()