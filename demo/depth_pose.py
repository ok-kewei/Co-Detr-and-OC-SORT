import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_3d_point_cloud(X, Y, Z, downsample_rate=1, cmap='viridis'):
    """
    Visualize 3D coordinates as a point cloud with X as the horizontal axis,
    Y as the vertical axis, and Z as the depth (distance from the camera).

    Parameters:
    - X, Y, Z (np.ndarray): 2D arrays containing the X, Y, and Z coordinates of each point.
    - downsample_rate (int): Factor by which to downsample the points for faster visualization.
    - cmap (str): Color map for depth-based coloring (default: 'viridis').
    """

    # Downsample if needed
    if downsample_rate > 1:
        X = X[::downsample_rate, ::downsample_rate]
        Y = Y[::downsample_rate, ::downsample_rate]
        Z = Z[::downsample_rate, ::downsample_rate]

    # Flatten the arrays for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the 3D points
    sc = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap=cmap, marker='o', s=0.5)

    # Add a color bar for depth indication
    cbar = plt.colorbar(sc, ax=ax, label='Depth (Z)')

    # Set axis labels
    ax.set_xlabel("X (Horizontal, meters)")
    ax.set_ylabel("Y (Vertical, meters)")
    ax.set_zlabel("Z (Depth, meters)")

    # Optional: set limits if you have an expected range
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(0, 4)

    plt.show()



# Replace 'file_path.npy' with the path to your .npy file
file_path = '1646637711683303680_disp.npy'
Z = np.load(file_path)

# Now 'data' contains the contents of the .npy file as a NumPy array
print(Z)

# Camera intrinsic parameters
fx, fy = 1190.326736034756/2,1194.0894306354906/2 # Example focal lengths in pixels
cx, cy = 1011.0395713062037/2,544.9116971226719/2 # Example principal point

# Get the shape of the depth map
height, width = Z.shape

# Generate a grid of (u, v) coordinates
u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

# def pixel_to_3d(u, v, z, fx, fy, cx, cy):
#     # Convert pixel coordinates and depth to 3D camera coordinates
#     X = (u - cx) * z / fx
#     Y = (v - cy) * z / fy
#     Z = z
#     return X, Y, Z



# Compute 3D coordinates for every pixel
X = (u_coords - cx) * Z / fx
Y = (v_coords - cy) * Z / fy
Z = Z  # Z is already the depth value

visualize_3d_point_cloud(X, Y, Z, downsample_rate=2)

# Calculate 3D coordinates
# X, Y, Z = pixel_to_3d(u, v, z, fx, fy, cx, cy)
print("3D coordinates:", X, Y, Z)