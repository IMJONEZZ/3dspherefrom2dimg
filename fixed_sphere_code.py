import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D

class SphereModel(nn.Module):
    """
    Differentiable sphere model with parameters for radius, stretch, rotation, and translation
    """
    def __init__(self, init_params=None):
        super(SphereModel, self).__init__()
        
        # Initialize with default parameters if none provided
        if init_params is None:
            # Default parameters
            radius = 1.0
            stretch = [1.0, 1.0, 1.0]  # sx, sy, sz
            rotation = [0.0, 0.0, 0.0]  # αx, αy, αz in radians
            translation = [0.0, 0.0, 5.0]  # tx, ty, tz
        else:
            radius, stretch, rotation, translation = init_params
            
        # Define learnable parameters
        self.radius = nn.Parameter(torch.tensor(radius, dtype=torch.float32))
        self.stretch = nn.Parameter(torch.tensor(stretch, dtype=torch.float32))
        self.rotation = nn.Parameter(torch.tensor(rotation, dtype=torch.float32))
        self.translation = nn.Parameter(torch.tensor(translation, dtype=torch.float32))
        
    def get_parameters(self):
        """Returns the current parameters of the sphere"""
        return {
            'radius': self.radius.item(),
            'stretch': self.stretch.detach().cpu().numpy(),
            'rotation': self.rotation.detach().cpu().numpy(),
            'translation': self.translation.detach().cpu().numpy()
        }
        
    def rotation_matrices(self):
        """Compute rotation matrices for x, y, z rotations"""
        # Rotation around X-axis
        cos_x, sin_x = torch.cos(self.rotation[0]), torch.sin(self.rotation[0])
        R_x = torch.stack([
            torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)]),
            torch.stack([torch.tensor(0.0), cos_x, -sin_x]),
            torch.stack([torch.tensor(0.0), sin_x, cos_x])
        ])
        
        # Rotation around Y-axis
        cos_y, sin_y = torch.cos(self.rotation[1]), torch.sin(self.rotation[1])
        R_y = torch.stack([
            torch.stack([cos_y, torch.tensor(0.0), sin_y]),
            torch.stack([torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)]),
            torch.stack([-sin_y, torch.tensor(0.0), cos_y])
        ])
        
        # Rotation around Z-axis
        cos_z, sin_z = torch.cos(self.rotation[2]), torch.sin(self.rotation[2])
        R_z = torch.stack([
            torch.stack([cos_z, -sin_z, torch.tensor(0.0)]),
            torch.stack([sin_z, cos_z, torch.tensor(0.0)]),
            torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])
        ])
        
        # Combined rotation matrix (Z * Y * X order)
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
        return R
    
    def forward(self, points_3d):
        """
        Transform 3D points using current sphere parameters
        
        Args:
            points_3d: Tensor of 3D points [N, 3]
            
        Returns:
            Transformed 3D points
        """
        # Apply stretch
        stretched_points = points_3d * self.stretch
        
        # Apply rotation
        R = self.rotation_matrices()
        rotated_points = torch.matmul(stretched_points, R.T)
        
        # Apply radius and translation
        transformed_points = self.radius * rotated_points + self.translation
        
        return transformed_points

class DifferentiableRenderer:
    """
    Fully differentiable renderer to project 3D sphere to 2D image
    """
    def __init__(self, image_size=(512, 512), camera_params=None):
        self.image_size = image_size
        
        # Default camera parameters if none provided
        if camera_params is None:
            self.focal_length = 500.0  # focal length
            self.principal_point = (image_size[0] / 2, image_size[1] / 2)  # principal point (cx, cy)
        else:
            self.focal_length = camera_params['focal_length']
            self.principal_point = camera_params['principal_point']
            
        # Create coordinate grid for differentiable rendering
        self.create_coordinate_grid()
        
    def create_coordinate_grid(self):
        """Create pixel coordinate grid for the entire image"""
        height, width = self.image_size
        y_coords = torch.arange(0, height, dtype=torch.float32)
        x_coords = torch.arange(0, width, dtype=torch.float32)
        self.y_grid, self.x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
    def project_3d_to_2d(self, points_3d):
        """
        Project 3D points to 2D using perspective projection
        
        Args:
            points_3d: Tensor of 3D points [N, 3]
            
        Returns:
            Tensor of 2D points [N, 2]
        """
        # Simple perspective projection: (x,y,z) -> (fx*x/z + cx, fy*y/z + cy)
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # Avoid division by zero
        z = torch.clamp(z, min=1e-5)
        
        # Project to 2D
        u = self.focal_length * x / z + self.principal_point[0]
        v = self.focal_length * y / z + self.principal_point[1]
        
        return torch.stack([u, v], dim=1)
        
    def render_silhouette(self, sphere_model, resolution=100):
        """
        Render a silhouette of the sphere using a fully differentiable approach
        
        Args:
            sphere_model: SphereModel instance
            resolution: Resolution of the sphere sampling
            
        Returns:
            Silhouette image as tensor
        """
        print("Rendering Silhouette")
        # Create a grid of points on a unit sphere
        u = torch.linspace(0, 2 * np.pi, resolution)
        v = torch.linspace(0, np.pi, resolution)
        
        
        # Create meshgrid of sphere points
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        
        # Convert to Cartesian coordinates (points on a unit sphere)
        x = torch.sin(v_grid) * torch.cos(u_grid)
        y = torch.sin(v_grid) * torch.sin(u_grid)
        z = torch.cos(v_grid)
        
        # Combine into points tensor [N, 3]
        points_3d = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Transform points using sphere model
        transformed_points = sphere_model(points_3d)
        
        # Project to 2D
        points_2d = self.project_3d_to_2d(transformed_points)
        
        # Create a differentiable silhouette using soft rendering
        height, width = self.image_size
        silhouette = torch.zeros((height, width), dtype=torch.float32)
       
        # Convert to device of input tensors
        x_grid = self.x_grid.to(points_2d.device)
        y_grid = self.y_grid.to(points_2d.device)
        
        # Use Gaussian kernel to create soft silhouette
        # For each point, add a Gaussian blob to the silhouette
        sigma = 2.0  # Controls the spread of the Gaussian
        print(f"Using a Gaussian Kernel to create a soft silhouette in range: {len(range(points_2d.shape[0]))}")
        for i in range(points_2d.shape[0]):
            point_x, point_y = points_2d[i, 0], points_2d[i, 1]
            
            # Skip points that are far outside the image
            if (point_x < -width/2 or point_x > width*1.5 or 
                point_y < -height/2 or point_y > height*1.5):
                continue
            
            # Calculate squared distance to each pixel (vectorized)
            squared_dist = (x_grid - point_x)**2 + (y_grid - point_y)**2
            
            # Apply Gaussian kernel and add to silhouette
            gaussian = torch.exp(-squared_dist / (2 * sigma**2))
            silhouette += gaussian
       
        # Normalize silhouette to [0, 1] range
        silhouette = torch.clamp(silhouette / torch.max(silhouette), 0, 1)
        
        return silhouette

def extract_silhouette(image, threshold=127):
    """
    Extract binary silhouette from an image
    
    Args:
        image: Input image
        threshold: Threshold value for binarization
        
    Returns:
        Binary silhouette
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold to get binary mask
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Normalize to [0, 1]
    silhouette = binary / 255.0
    
    return silhouette

def initialize_sphere_params(silhouette):
    """
    Initialize sphere parameters from silhouette using simple heuristics
    
    Args:
        silhouette: Binary silhouette image
        
    Returns:
        Dictionary of initial parameters
    """
    # Find contours in the silhouette
    contours = measure.find_contours(silhouette, 0.5)
    
    if len(contours) == 0:
        # Default parameters if no contours found
        return {
            'radius': 1.0,
            'stretch': [1.0, 1.0, 1.0],
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, 0.0, 5.0]
        }
    
    # Use the largest contour
    contour = sorted(contours, key=lambda x: len(x))[-1]
    
    # Calculate center and axes from contour
    y_coords, x_coords = contour[:, 0], contour[:, 1]
    center_y, center_x = np.mean(y_coords), np.mean(x_coords)
    
    # Estimate radius from contour (average distance from center to points)
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    radius_estimate = np.mean(distances)
    
    # Estimate stretch factors using PCA on contour points
    points = np.column_stack([x_coords - center_x, y_coords - center_y])
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Convert eigenvalues to stretch factors (normalized)
    stretch_factors = np.sqrt(eigenvalues) / np.sqrt(np.max(eigenvalues))
    if len(stretch_factors) < 2:
        stretch_factors = np.ones(2)
    
    # Complete with z-stretch (assumed to be similar to smaller of x,y)
    stretch_z = min(stretch_factors)
    stretch_factors = [stretch_factors[0], stretch_factors[1], stretch_z]
    
    # Estimate rotation from eigenvectors
    # This is a simple approximation that works for some cases
    rotation_angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    
    # Initial parameters
    init_params = {
        'radius': radius_estimate / 100.0,  # Scale down for numerical stability
        'stretch': stretch_factors,
        'rotation': [0.0, 0.0, rotation_angle],
        'translation': [center_x - silhouette.shape[1]/2, 
                        center_y - silhouette.shape[0]/2, 
                        5.0]  # Assuming z is 5 units away
    }
    
    return init_params

def visualize_results(original_image, original_silhouette, rendered_silhouette, 
                      params, gt_params=None, iteration=None):
    """
    Visualize the fitting results
    
    Args:
        original_image: Original input image
        original_silhouette: Ground truth silhouette
        rendered_silhouette: Rendered silhouette from model
        params: Estimated parameters
        gt_params: Ground truth parameters (optional)
        iteration: Current iteration number (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth silhouette
    axes[1].imshow(original_silhouette, cmap='gray')
    axes[1].set_title('Ground Truth Silhouette')
    axes[1].axis('off')
    
    # Rendered silhouette
    axes[2].imshow(rendered_silhouette.detach().cpu().numpy(), cmap='gray')
    if iteration is not None:
        axes[2].set_title(f'Rendered Silhouette (Iteration {iteration})')
    else:
        axes[2].set_title('Rendered Silhouette')
    axes[2].axis('off')
    
    # Add text with parameters
    param_text = f"Radius: {params['radius']:.4f}\n"
    param_text += f"Stretch: [{params['stretch'][0]:.4f}, {params['stretch'][1]:.4f}, {params['stretch'][2]:.4f}]\n"
    param_text += f"Rotation: [{params['rotation'][0]:.4f}, {params['rotation'][1]:.4f}, {params['rotation'][2]:.4f}]\n"
    param_text += f"Translation: [{params['translation'][0]:.4f}, {params['translation'][1]:.4f}, {params['translation'][2]:.4f}]"
    
    plt.figtext(0.5, 0.01, param_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # If ground truth parameters are provided, display them for comparison
    if gt_params is not None:
        gt_text = f"GT Radius: {gt_params['radius']:.4f}\n"
        gt_text += f"GT Stretch: [{gt_params['stretch'][0]:.4f}, {gt_params['stretch'][1]:.4f}, {gt_params['stretch'][2]:.4f}]\n"
        gt_text += f"GT Rotation: [{gt_params['rotation'][0]:.4f}, {gt_params['rotation'][1]:.4f}, {gt_params['rotation'][2]:.4f}]\n"
        gt_text += f"GT Translation: [{gt_params['translation'][0]:.4f}, {gt_params['translation'][1]:.4f}, {gt_params['translation'][2]:.4f}]"
        
        plt.figtext(0.5, 0.15, gt_text, ha='center', fontsize=10, 
                    bbox=dict(facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_3d_sphere(params, gt_params=None):
    """
    Visualize the 3D sphere with estimated parameters
    
    Args:
        params: Estimated parameters
        gt_params: Ground truth parameters (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Helper function to transform points based on parameters
    def transform_points(points, params):
        # Extract parameters
        radius = params['radius']
        sx, sy, sz = params['stretch']
        rx, ry, rz = params['rotation']
        tx, ty, tz = params['translation']
        
        # Apply stretch
        points_stretched = points.copy()
        points_stretched[0] *= sx
        points_stretched[1] *= sy
        points_stretched[2] *= sz
        
        # Apply rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        R = np.matmul(Rz, np.matmul(Ry, Rx))
        
        points_rotated = np.zeros_like(points_stretched)
        for i in range(points_stretched.shape[1]):
            for j in range(points_stretched.shape[2]):
                points_rotated[:, i, j] = np.matmul(R, points_stretched[:, i, j])
        
        # Apply radius and translation
        points_transformed = radius * points_rotated
        points_transformed[0] += tx
        points_transformed[1] += ty
        points_transformed[2] += tz
        
        return points_transformed
    
    # Transform points for estimated parameters
    points = np.array([x, y, z])
    points_transformed = transform_points(points, params)
    
    # Plot estimated sphere
    ax.plot_surface(points_transformed[0], points_transformed[1], points_transformed[2], 
                    color='b', alpha=0.3, label='Estimated')
    
    # If ground truth is provided, also plot it
    if gt_params is not None:
        points_gt = transform_points(points, gt_params)
        ax.plot_surface(points_gt[0], points_gt[1], points_gt[2], 
                        color='g', alpha=0.3, label='Ground Truth')
    
    # Plot coordinate axes
    axis_length = max(params['radius'] * 2, 1.0)
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', label='X-axis')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', label='Z-axis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Sphere Visualization')
    
    # Set equal scaling for all axes
    max_range = np.array([
        np.max(points_transformed[0]) - np.min(points_transformed[0]),
        np.max(points_transformed[1]) - np.min(points_transformed[1]),
        np.max(points_transformed[2]) - np.min(points_transformed[2])
    ]).max() / 2.0
    
    mid_x = (np.max(points_transformed[0]) + np.min(points_transformed[0])) * 0.5
    mid_y = (np.max(points_transformed[1]) + np.min(points_transformed[1])) * 0.5
    mid_z = (np.max(points_transformed[2]) + np.min(points_transformed[2])) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def optimize_sphere_parameters(image, silhouette, camera_params=None, 
                               num_iterations=100, learning_rate=0.01, 
                               visualize_interval=10, gt_params=None):
    """
    Optimize sphere parameters to match the silhouette
    
    Args:
        image: Original input image
        silhouette: Ground truth silhouette
        camera_params: Camera parameters
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimizer
        visualize_interval: Interval for visualization
        gt_params: Ground truth parameters (optional)
        
    Returns:
        Dictionary of optimized parameters
    """
    # Initialize parameters
    print("Initializing Sphere Params around silhouette")
    init_params_dict = initialize_sphere_params(silhouette)
    init_params = [
        init_params_dict['radius'],
        init_params_dict['stretch'],
        init_params_dict['rotation'],
        init_params_dict['translation']
    ]
    
    # Create sphere model and renderer
    sphere_model = SphereModel(init_params)
    renderer = DifferentiableRenderer(image_size=silhouette.shape, camera_params=camera_params)
    
    # Convert silhouette to tensor
    silhouette_tensor = torch.tensor(silhouette, dtype=torch.float32)
    
    # Define optimizer
    optimizer = optim.Adam(sphere_model.parameters(), lr=learning_rate)
    
    # Define loss function (MSE for smoother optimization)
    loss_fn = nn.MSELoss()
    
    # Track losses
    losses = []
    
    # Optimization loop
    for iteration in range(num_iterations):
        # Zero gradients
        optimizer.zero_grad()
        
        # Render silhouette
        rendered_silhouette = renderer.render_silhouette(sphere_model)
        
        # Compute loss
        loss = loss_fn(rendered_silhouette, silhouette_tensor)
        losses.append(loss.item())
        
        # Backpropagate
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")
        
        # Visualize current result
        if (iteration + 1) % visualize_interval == 0 or iteration == 0:
            params = sphere_model.get_parameters()
            visualize_results(image, silhouette, rendered_silhouette, 
                             params, gt_params, iteration + 1)
    
    # Final parameters
    final_params = sphere_model.get_parameters()
    print(f"Final Params: {final_params}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Loss')
    plt.grid(True)
    plt.show()
    
    # Visualize final result
    rendered_silhouette = renderer.render_silhouette(sphere_model)
    visualize_results(image, silhouette, rendered_silhouette, final_params, gt_params)
    
    # Visualize 3D sphere
    visualize_3d_sphere(final_params, gt_params)
    
    return final_params

def load_test_data(image_path, camera_params_path=None, gt_params_path=None):
    """
    Load test data including image, camera parameters and ground truth
    
    Args:
        image_path: Path to input image
        camera_params_path: Path to camera parameters (optional)
        gt_params_path: Path to ground truth parameters (optional)
        
    Returns:
        Tuple of (image, camera_params, gt_params)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        # If loading fails, generate a synthetic test image
        print("Image not found. Generating synthetic test image.")
        image = generate_synthetic_test_data()
    
    # Load camera parameters if provided
    camera_params = None
    if camera_params_path is not None:
        try:
            # Implement camera parameter loading based on file format
            pass
        except:
            print("Failed to load camera parameters.")
    
    # Load ground truth parameters if provided
    gt_params = None
    if gt_params_path is not None:
        try:
            # Implement ground truth parameter loading based on file format
            pass
        except:
            print("Failed to load ground truth parameters.")
    
    return image, camera_params, gt_params

def generate_synthetic_test_data(image_size=(512, 512)):
    """
    Generate synthetic test data for debugging
    
    Args:
        image_size: Size of the synthetic image
        
    Returns:
        Synthetic image of a sphere
    """
    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Define sphere parameters
    center = (image_size[0] // 2, image_size[1] // 2)
    radius = min(image_size) // 4
    
    # Draw a filled circle
    cv2.circle(image, center, radius, (0, 0, 255), -1)
    
    # Add shading to make it look 3D
    for y in range(image_size[0]):
        for x in range(image_size[1]):
            # Distance from the center
            dx = x - center[0]
            dy = y - center[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < radius:
                # Calculate z coordinate on the sphere surface
                z = np.sqrt(radius**2 - dx**2 - dy**2)
                
                # Simple shading based on surface normal
                light_dir = np.array([0.5, 0.5, 1.0])
                light_dir = light_dir / np.linalg.norm(light_dir)
                
                normal = np.array([dx, dy, z])
                normal = normal / np.linalg.norm(normal)
                
                # Dot product for diffuse shading
                shading = np.dot(normal, light_dir)
                shading = max(0.2, min(1.0, shading))  # Clamp between 0.2 and 1.0
                
                # Apply shading
                image[y, x, 0] = int(255 * shading)
                image[y, x, 1] = int(100 * shading)
                image[y, x, 2] = int(100 * shading)
    
    return image

def main(image_path=None, camera_params_path=None, gt_params_path=None):
    """
    Main function to run the sphere parameter estimation
    
    Args:
        image_path: Path to input image
        camera_params_path: Path to camera parameters (optional)
        gt_params_path: Path to ground truth parameters (optional)
    """
    # Load or generate test data
    if image_path is None:
        print("image_path not given, generating synthetic")
        # Generate synthetic test data
        image = generate_synthetic_test_data()
        camera_params = None
        
        # Define known ground truth parameters for the synthetic data
        gt_params = {
            'radius': 1.0,
            'stretch': [1.0, 1.0, 1.0],
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, 0.0, 5.0]
        }
        print(f"Synthetic Image: {image}\nSynthetic Ground Truth: {gt_params}")
    else:
        print(f"loading image from path: {image_path}")
        # Load test data
        image, camera_params, gt_params = load_test_data(
            image_path, camera_params_path, gt_params_path)
    
    # Extract silhouette from image
    print("extracting silhouette")
    silhouette = extract_silhouette(image)
    
    # Optimize sphere parameters
    print(f"optimizing sphere parameters based on: {silhouette}")
    optimized_params = optimize_sphere_parameters(
        image, silhouette, camera_params, 
        num_iterations=50, 
        learning_rate=0.01,
        visualize_interval=10,
        gt_params=gt_params
    )
    
    print("Optimization complete!")
    print("Final parameters:")
    print(f"Radius: {optimized_params['radius']:.4f}")
    print(f"Stretch: {optimized_params['stretch']}")
    print(f"Rotation: {optimized_params['rotation']}")
    print(f"Translation: {optimized_params['translation']}")
    
    if gt_params is not None:
        print("\nGround truth parameters:")
        print(f"Radius: {gt_params['radius']:.4f}")
        print(f"Stretch: {gt_params['stretch']}")
        print(f"Rotation: {gt_params['rotation']}")
        print(f"Translation: {gt_params['translation']}")

if __name__ == "__main__":
    main(image_path="./example_sphere.png")
