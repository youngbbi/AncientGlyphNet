from PIL import Image, ImageFilter
import numpy as np

# Reload the original image to start fresh
image_path = 'D:\docu\Lab\毕业需要的东西\小论文\图\\1706341371275.jpg'
image = Image.open(image_path)
image_array = np.array(image)

# Check if the image has an alpha channel (transparency)
has_alpha = image_array.shape[2] == 4

# If there is an alpha channel, separate it from the color channels
if has_alpha:
    color_channels = image_array[:, :, :3].copy()
    alpha_channel = image_array[:, :, 3].copy()
else:
    color_channels = image_array.copy()

# Define colors for deep red and blue
deep_red = np.array([139, 0, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)

# Create a mask for the red elements
red_threshold = {
    'red_min': 100,
    'green_max': 50,
    'blue_max': 50
}
red_mask = (color_channels[:, :, 0] > red_threshold['red_min']) & \
           (color_channels[:, :, 1] < red_threshold['green_max']) & \
           (color_channels[:, :, 2] < red_threshold['blue_max'])

# Perform closing to fill in the gaps between red elements
binary_image = red_mask.astype(np.uint8) * 255
closed_image = Image.fromarray(binary_image).filter(ImageFilter.MaxFilter(size=5))
closed_mask = np.array(closed_image) > 0

# Apply the deep red color to the closed mask areas
color_channels[closed_mask] = deep_red

# Apply blue to all other areas
color_channels[~closed_mask] = blue

# Reassemble the final image including the alpha channel if present
if has_alpha:
    final_image_array = np.dstack((color_channels, alpha_channel))
else:
    final_image_array = color_channels

# Convert the numpy array to uint8 type
final_image_array = final_image_array.astype(np.uint8)

# Convert the numpy array back to an image
final_image_with_correct_type = Image.fromarray(final_image_array)

# Save the new image
final_corrected_image_path = 'D:\docu\Lab\毕业需要的东西\小论文\图\\final_corrected_colored_probability_map.png'
final_image_with_correct_type.save(final_corrected_image_path)
