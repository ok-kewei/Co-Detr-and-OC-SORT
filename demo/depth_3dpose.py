import numpy as np
import cv2


def compute_3d_pose(disparity_map, focal_length, baseline, principal_point):
    """
    Convert disparity map to 3D coordinates.

    Args:
        disparity_map (numpy.ndarray): Disparity map (grayscale image, values in pixels).
        focal_length (float): Focal length of the camera in pixels.
        baseline (float): Baseline distance between the two cameras in meters.
        principal_point (tuple): (c_x, c_y) coordinates of the principal point (optical center).

    Returns:
        numpy.ndarray: 3D coordinates (X, Y, Z) of each pixel in the disparity map.
    """

    # Get the shape of the disparity map
    height, width = disparity_map.shape

    # Create an empty array to store the 3D coordinates
    points_3d = np.zeros((height, width, 3), dtype=np.float32)

    # Principal point coordinates (optical center)
    c_x, c_y = principal_point

    # Loop over every pixel in the disparity map
    for y in range(height):
        for x in range(width):
            disparity = disparity_map[y, x]
            if disparity > 0:  # Discard points where disparity is zero (invalid points)
                # Calculate depth Z using the disparity formula
                Z = (focal_length * baseline) / disparity

                # Calculate 3D coordinates (X, Y, Z)
                X = (x - c_x) * Z / focal_length
                Y = (y - c_y) * Z / focal_length

                # Store the 3D coordinates
                points_3d[y, x] = [X, Y, Z]

    return points_3d


# Example usage
if __name__ == "__main__":
    # Load a disparity map (example: you can replace it with your own disparity map)
    disparity_map = cv2.imread("disparity_map.png", cv2.IMREAD_GRAYSCALE)

    # Camera parameters (these should come from your camera calibration)
    focal_length = 1000  # Focal length in pixels (example value)
    baseline = 0.1  # Baseline in meters (example value)
    principal_point = (
    disparity_map.shape[1] // 2, disparity_map.shape[0] // 2)  # Assuming the principal point is at the center

    # Compute 3D pose (coordinates) from disparity
    points_3d = compute_3d_pose(disparity_map, focal_length, baseline, principal_point)

    # You can now use the points_3d for visualization or further processing
    print("3D Points Shape:", points_3d.shape)
    print("Example 3D Point at (100, 100):", points_3d[100, 100])  # Example for one point