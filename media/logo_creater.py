import numpy as np
import math

import matplotlib.pyplot as plt


def draw_circle(width, height):
    # Create a blank image with the specified dimensions
    image = np.ones((height, width))

    center_x = width // 2
    center_y = height // 2

    radius = min(center_x, center_y)

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Calculate the distance from the current pixel to the center of the circle
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # If the distance is less than or equal to the radius, set the pixel value to 0 (black)
            if distance <= radius:
                image[y, x] = 0

    return image


def add_gradient(image, custom_height, gradient_range, start_y, stop_y):
    height, width = image.shape

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Calculate the probability of setting the pixel to white based on y value
            probability = ((y - start_y) / (stop_y - start_y)) * gradient_range

            # Generate a random number between 0 and 1
            random_number = np.random.rand()

            # If the random number is less than or equal to the probability and y is between start_y and stop_y, set the pixel value to 1 (white)
            if random_number <= probability and start_y <= y <= stop_y:
                image[y, x] = 1

    return image


def set_left_to_diagonal_white(image, angle, x_offset):
    height, width = image.shape

    # Calculate the slope of the diagonal line based on the angle
    slope = math.tan(math.radians(angle))

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Calculate the y coordinate on the diagonal line based on the x coordinate and slope
            diagonal_y = int(slope * (x - x_offset))

            # If the y coordinate is greater than or equal to the calculated diagonal y coordinate, set the pixel value to 1 (white)
            if y >= diagonal_y:
                image[y, x] = 1

    return image


# # Plot the image
# image = draw_circle(50, 50)
# image_with_gradient = add_gradient(image, custom_height=25, gradient_range=1, start_y=10, stop_y=50)
# image_with_gradient = set_left_to_diagonal_white(image_with_gradient, angle=65, x_offset=15)
# plt.imshow(image_with_gradient, cmap='gray')
# plt.show()

# image = draw_circle(35, 35)
# image_with_gradient = add_gradient(image, custom_height=25, gradient_range=1, start_y=10, stop_y=50)
# image_with_gradient = set_left_to_diagonal_white(image_with_gradient, angle=65, x_offset=7)
# plt.imshow(image_with_gradient, cmap='gray')
# plt.show()

# image = draw_circle(40, 40)
# image_with_gradient = add_gradient(image, custom_height=25, gradient_range=1, start_y=15, stop_y=50)
# image_with_gradient = set_left_to_diagonal_white(image_with_gradient, angle=50, x_offset=0)
# plt.imshow(image_with_gradient, cmap='gray')
# plt.show()

# Plot the image
image = draw_circle(50, 50)
image_with_gradient = add_gradient(image, custom_height=25, gradient_range=1, start_y=10, stop_y=50)
image_with_gradient = set_left_to_diagonal_white(image_with_gradient, angle=70, x_offset=15)
plt.imshow(image_with_gradient, cmap='gray')
plt.show()

# save as png
# plt.imsave('C:/Users/ansel/Downloads/circle.png', image_with_gradient, cmap='gray', dpi = 1000)
