import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-1, 1, 1000)

# Compute Runge function
y_runge = 1 / (1 + 25 * x**2)

# Plot the Runge function
plt.figure()
plt.plot(x, y_runge, label="Runge: f(x)=1/(1+25x^2)", color='#A10000', linewidth=1.5)
plt.title("Runge Function", fontsize=16, fontfamily='sans-serif', pad=10)
plt.xlabel("x", fontsize=12, fontfamily='sans-serif')
plt.ylabel("y", fontsize=12, fontfamily='sans-serif')
plt.legend(fontsize=8, loc='upper right')
fig = plt.gcf()
fig.set_size_inches(6, 4)   # <- force size here
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("Plots/runge_function.pdf", bbox_inches="tight")
plt.show()
