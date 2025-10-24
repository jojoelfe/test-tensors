"""Visualization functions for 3D tensors."""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import trimesh


def visualize_3d_tensor(
    volume: np.ndarray,
    title: str = "3D Tensor Visualization",
    figsize: tuple[int, int] = (15, 10),
    threshold: float = 0.5,
    show_3d: bool = True,
    pitch: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Visualize a 3D tensor with multiple views and 3D isosurface.

    Creates a comprehensive visualization showing:
    - Central slices along all three dimensions (XY, XZ, YZ)
    - Maximum intensity projections along all three dimensions
    - 3D isosurface rendering using trimesh

    Parameters
    ----------
    volume : np.ndarray
        3D numpy array to visualize. Should have shape (Z, Y, X).
    title : str, optional
        Title for the visualization. Default is "3D Tensor Visualization".
    figsize : tuple of int, optional
        Figure size for matplotlib plots as (width, height). Default is (15, 10).
    threshold : float, optional
        Threshold value for isosurface generation. Default is 0.5.
    show_3d : bool, optional
        Whether to show the 3D isosurface viewer. Default is True.
    pitch : tuple of float, optional
        Voxel size in (Z, Y, X) dimensions for 3D rendering. Default is (1.0, 1.0, 1.0).

    Raises
    ------
    ImportError
        If required visualization dependencies are not installed.
    ValueError
        If volume is not 3D or has invalid dimensions.

    Examples
    --------
    >>> from test_tensors import generate_cross_3d, visualize_3d_tensor
    >>> cross = generate_cross_3d(32)
    >>> visualize_3d_tensor(cross, title="Cross Pattern")

    Notes
    -----
    This function requires optional dependencies that can be installed with:
    pip install test-tensors[viz]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = (
            "Visualization requires matplotlib. "
            "Install with: pip install test-tensors[viz]"
        )
        raise ImportError(msg) from e

    if volume.ndim != 3:
        msg = f"Expected 3D array, got {volume.ndim}D"
        raise ValueError(msg)

    if volume.size == 0:
        msg = "Volume cannot be empty"
        raise ValueError(msg)

    # Get volume dimensions
    nz, ny, nx = volume.shape
    center_z, center_y, center_x = nz // 2, ny // 2, nx // 2

    # Create matplotlib figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Central slices
    axes[0, 0].imshow(volume[center_z, :, :], cmap="viridis", origin="lower")
    axes[0, 0].set_title(f"XY Slice (Z={center_z})")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")

    axes[0, 1].imshow(volume[:, center_y, :], cmap="viridis", origin="lower")
    axes[0, 1].set_title(f"XZ Slice (Y={center_y})")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Z")

    axes[0, 2].imshow(volume[:, :, center_x], cmap="viridis", origin="lower")
    axes[0, 2].set_title(f"YZ Slice (X={center_x})")
    axes[0, 2].set_xlabel("Y")
    axes[0, 2].set_ylabel("Z")

    # Maximum intensity projections
    axes[1, 0].imshow(np.max(volume, axis=0), cmap="viridis", origin="lower")
    axes[1, 0].set_title("XY Projection (Max Z)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")

    axes[1, 1].imshow(np.max(volume, axis=1), cmap="viridis", origin="lower")
    axes[1, 1].set_title("XZ Projection (Max Y)")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Z")

    axes[1, 2].imshow(np.max(volume, axis=2), cmap="viridis", origin="lower")
    axes[1, 2].set_title("YZ Projection (Max X)")
    axes[1, 2].set_xlabel("Y")
    axes[1, 2].set_ylabel("Z")

    plt.tight_layout()
    plt.show()

    # 3D isosurface visualization
    if show_3d:
        _show_3d_isosurface(volume, threshold=threshold, pitch=pitch, title=title)


def _show_3d_isosurface(
    volume: np.ndarray,
    threshold: float = 0.5,
    pitch: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "3D Isosurface",
) -> None:
    """Create and display 3D isosurface using trimesh.

    Parameters
    ----------
    volume : np.ndarray
        3D numpy array to visualize.
    threshold : float
        Threshold value for isosurface generation.
    pitch : tuple of float
        Voxel size in (Z, Y, X) dimensions.
    title : str
        Title for the 3D viewer window.
    """
    try:
        import trimesh
        from skimage import measure
    except ImportError as e:
        msg = (
            "3D visualization requires trimesh and scikit-image. "
            "Install with: pip install test-tensors[viz]"
        )
        raise ImportError(msg) from e

    # Generate isosurface using marching cubes
    try:
        # Use scikit-image's marching cubes for better control
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=threshold, spacing=pitch
        )

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Set mesh color (blue)
        mesh.visual.face_colors = np.array([100, 150, 255, 255], dtype=np.uint8)

        # Create scene with coordinate axes
        scene = trimesh.Scene([mesh])

        # Add coordinate system
        # Scale axes based on volume size
        max_dim = max(volume.shape)
        axis_length = max_dim * 0.3
        axis_radius = max_dim * 0.01

        coordinate_axes = trimesh.creation.axis(
            origin_size=axis_radius * 2,
            axis_radius=axis_radius,
            axis_length=axis_length,
        )
        scene.add_geometry(coordinate_axes)

        # Show the scene
        scene.show(caption=title)

    except Exception as e:
        print(f"Warning: Could not generate 3D visualization: {e}")
        print("This might happen if the volume has no surfaces above the threshold.")


def create_demo_notebook() -> str:
    """Create a demonstration notebook showing visualization capabilities.

    Returns
    -------
    str
        Path to the created notebook file.
    """
    notebook_content = '''````xml
<VSCode.Cell language="markdown">
# 3D Tensor Visualization Demo

This notebook demonstrates the visualization capabilities of the test-tensors library.
</VSCode.Cell>
<VSCode.Cell language="python">
# Install visualization dependencies if needed
# !pip install test-tensors[viz]

import numpy as np
from test_tensors import generate_cross_3d, visualize_3d_tensor
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Generate a 3D Cross Pattern

First, let's create a 3D cross pattern tensor.
</VSCode.Cell>
<VSCode.Cell language="python">
# Generate a 3D cross pattern
cross = generate_cross_3d(32)
print(f"Shape: {cross.shape}")
print(f"Data type: {cross.dtype}")
print(f"Value range: {cross.min():.1f} to {cross.max():.1f}")
print(f"Non-zero elements: {np.count_nonzero(cross)}")
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Visualize the Cross Pattern

Now let's visualize the cross pattern with multiple views and 3D rendering.
</VSCode.Cell>
<VSCode.Cell language="python">
# Visualize the cross pattern
visualize_3d_tensor(
    cross, 
    title="3D Cross Pattern Visualization",
    threshold=0.5,
    show_3d=True
)
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Custom Rectangular Cross

Let's try a rectangular volume to see how the visualization adapts.
</VSCode.Cell>
<VSCode.Cell language="python">
# Generate rectangular cross
rect_cross = generate_cross_3d((20, 30, 40))
print(f"Rectangular shape: {rect_cross.shape}")

# Visualize with custom settings
visualize_3d_tensor(
    rect_cross,
    title="Rectangular Cross Pattern",
    figsize=(18, 12),
    threshold=0.5,
    pitch=(0.8, 1.0, 1.2)  # Different voxel sizes for each dimension
)
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Understanding the Views

The visualization shows:
- **Top row**: Central slices through each dimension (XY, XZ, YZ)
- **Bottom row**: Maximum intensity projections along each axis
- **3D viewer**: Interactive isosurface rendering with coordinate axes

The 3D viewer allows you to:
- Rotate by dragging
- Zoom with mouse wheel
- Pan by shift+drag
</VSCode.Cell>
````'''

    # Write to file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", delete=False, prefix="tensor_viz_demo_"
    ) as f:
        f.write(notebook_content)
        return f.name