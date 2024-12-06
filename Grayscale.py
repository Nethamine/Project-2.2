import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "foto_laser.png"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get the pixel values as a NumPy array
pixel_values = np.array(gray_image)

# Find the maximum pixel value in the image (excluding 255 for white pixels)
max_value = np.max(pixel_values[pixel_values < 255])

# Replace all maximum values in the array with 0 (or any placeholder)
pixel_values_cleaned = np.where(pixel_values == max_value, 0, pixel_values)

# Plot the grayscale pixel matrix
plt.figure(figsize=(10, 10))
plt.imshow(pixel_values_cleaned, cmap='gray')  # Use 'gray' colormap for grayscale image
plt.colorbar(label="Pixel Intensity")  # Add a color bar for intensity reference
plt.title(f"Grayscale Image with Max Value {max_value} Removed")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")



plt.tight_layout()
plt.show()

# Optionally save the modified grayscale image
cv2.imwrite("grayscale_image_cleaned.jpg", pixel_values_cleaned)
