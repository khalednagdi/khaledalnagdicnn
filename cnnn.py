import numpy as np

# Apply Convolution Function
def apply_convolution(image, filter, stride=1, padding=0):
    # Apply padding if needed
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Get dimensions
    image_size = image.shape[0]
    filter_size = filter.shape[0]

    # Ensure output_size is never negative
    output_size = max(1, (image_size - filter_size) // stride + 1)

    # Create output matrix
    output = np.zeros((output_size, output_size))

    # Perform convolution
    for i in range(output_size):
        for j in range(output_size):
            region = image[i * stride: i * stride + filter_size, j * stride: j * stride + filter_size]
            output[i, j] = np.sum(region * filter)

    return output

# ReLU Activation Function
def relu(matrix):
    return np.maximum(0, matrix)

# Max Pooling Function
def max_pooling(image, pool_size=2, stride=2):
    image_size = image.shape[0]
    output_size = max(1, (image_size - pool_size) // stride + 1)  # Ensure non-negative size
    output = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            region = image[i * stride: i * stride + pool_size, j * stride: j * stride + pool_size]
            output[i, j] = np.max(region)

    return output

# Sample 5x5 Image (Random Example)
image = np.array([
    [3, 0, 1, 2, 7],
    [1, 5, 8, 9, 3],
    [2, 7, 2, 5, 1],
    [0, 1, 3, 1, 7],
    [4, 2, 1, 6, 2]
])

# First Convolution Filter
conv_filter_1 = np.array([
    [1, 0, -1],
    [1, 1, -1],
    [-1, 0, 1]
])

# Apply First Convolution
R_conv1 = apply_convolution(image, conv_filter_1, stride=1, padding=0)
G_conv1 = apply_convolution(image, conv_filter_1, stride=1, padding=0)
B_conv1 = apply_convolution(image, conv_filter_1, stride=1, padding=0)

# Apply ReLU Activation
R_relu1 = relu(R_conv1)
G_relu1 = relu(G_conv1)
B_relu1 = relu(B_conv1)

# Apply First Max Pooling
R_pooled1 = max_pooling(R_relu1)
G_pooled1 = max_pooling(G_relu1)
B_pooled1 = max_pooling(B_relu1)

# Second Convolution Filter
conv_filter_2 = np.array([
    [1, -1, 0],
    [1, 1, -1],
    [-1, 0, 1]
])

# Apply Second Convolution
R_conv2 = apply_convolution(R_pooled1, conv_filter_2, stride=1, padding=0)
G_conv2 = apply_convolution(G_pooled1, conv_filter_2, stride=1, padding=0)
B_conv2 = apply_convolution(B_pooled1, conv_filter_2, stride=1, padding=0)

# Apply ReLU Activation Again
R_relu2 = relu(R_conv2)
G_relu2 = relu(G_conv2)
B_relu2 = relu(B_conv2)

# Apply Second Max Pooling
R_pooled2 = max_pooling(R_relu2)
G_pooled2 = max_pooling(G_relu2)
B_pooled2 = max_pooling(B_relu2)

# Compute Flatten Layer Size
flatten_size = R_pooled2.size + G_pooled2.size + B_pooled2.size

# Print Outputs
print("R channel after second max pooling:\n", R_pooled2)
print("G channel after second max pooling:\n", G_pooled2)
print("B channel after second max pooling:\n", B_pooled2)
print("Flatten layer size:", flatten_size)
