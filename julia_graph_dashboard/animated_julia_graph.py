import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate Julia set fractal
def julia_fractal(z, c, max_iter):
    for i in range(max_iter):
        if abs(z) > 2.0:
            return i
        z = z**2 + c
    return max_iter

# Function to generate Julia set image
def generate_julia_image(width, height, xmin, xmax, ymin, ymax, max_iter, c):
    image = np.zeros((height, width))

    for x in range(width):
        for y in range(height):
            zx = xmin + (x / (width - 1)) * (xmax - xmin)
            zy = ymin + (y / (height - 1)) * (ymax - ymin)

            value = julia_fractal(complex(zx, zy), c, max_iter)
            image[y, x] = value

    return image

# Function to update the plot for animation
def update(frame):
    ax.clear()
    global c_real
    c_real += 0.05
    c = complex(c_real, 0.27015)  # Increment the real part of the constant
    img = generate_julia_image(width, height, xmin, xmax, ymin, ymax, max_iter, c)
    im = ax.imshow(img, cmap='twilight_shifted')
    ax.set_title(f'Julia Set Fractal - Iteration: {frame}', color='black', fontname='Times New Roman')
    ax.set_xlabel('Real Axis', color='black', fontname='Times New Roman')
    ax.set_ylabel('Imaginary Axis', color='black', fontname='Times New Roman')
    ax.text(0.5, -0.15, f'Equation: $f(z) = z^2 + ({round(c.real,2)} + {round(c.imag,2)}i)$', transform=ax.transAxes, fontsize=10,
            color='black', fontname='Times New Roman', ha='center')

# Set parameters for the Julia set
width, height = 800, 800
xmin, xmax = -2, 2
ymin, ymax = -2, 2
max_iter = 50
c_real = -0.7  # Initial real part of the constant

# Create the initial plot
fig, ax = plt.subplots()
c = complex(c_real, 0.27015)
img = generate_julia_image(width, height, xmin, xmax, ymin, ymax, max_iter, c)
im = ax.imshow(img, cmap='twilight_shifted')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)

# Set up animation
animation = FuncAnimation(fig, update, frames=range(100), repeat=False, blit=False)
plt.show()
