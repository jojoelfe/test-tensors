"""Generate 3D test tensors for testing tensor manipulation algorithms."""


import numpy as np


def generate_cross_3d(shape: int | tuple[int, int, int] = 64) -> np.ndarray:
    """Generate a 3D tensor with a central cross pattern.

    Creates a 3D numpy array filled with zeros except for a cross pattern
    through the center of the volume. The cross extends through all three
    orthogonal planes (XY, XZ, YZ) passing through the center.

    Parameters
    ----------
    shape : int or tuple of int, optional
        Shape of the output tensor. If int, creates a cubic volume of size
        (shape, shape, shape). If tuple of 3 ints, creates volume with
        dimensions (shape[0], shape[1], shape[2]). Default is 64.

    Returns
    -------
    np.ndarray
        3D tensor of shape specified by input, with cross pattern set to 1.0
        and background set to 0.0. Array dtype is float64.

    Examples
    --------
    >>> # Create a 32x32x32 cubic volume with cross
    >>> cross = generate_cross_3d(32)
    >>> cross.shape
    (32, 32, 32)

    >>> # Create a rectangular volume with cross
    >>> cross = generate_cross_3d((64, 32, 16))
    >>> cross.shape
    (64, 32, 16)

    >>> # Check that center voxel is part of cross
    >>> cross = generate_cross_3d(10)
    >>> cross[5, 5, 5]
    1.0
    """
    # Parse shape parameter
    if isinstance(shape, int):
        dims = (shape, shape, shape)
    else:
        if len(shape) != 3:
            msg = "Shape must be int or tuple of 3 ints"
            raise ValueError(msg)
        dims = shape

    # Create empty volume
    volume = np.zeros(dims, dtype=np.float64)

    # Calculate center coordinates
    center_z, center_y, center_x = dims[0] // 2, dims[1] // 2, dims[2] // 2

    # Create cross pattern through center
    # XY plane cross (horizontal and vertical lines through center)
    volume[center_z, :, center_x] = 1.0  # Horizontal line in XY plane
    volume[center_z, center_y, :] = 1.0  # Vertical line in XY plane

    # XZ plane cross (lines through center)
    volume[:, center_y, center_x] = 1.0  # Line through Z direction

    # YZ plane cross (lines through center)
    volume[center_z, :, center_x] = 1.0  # Already set above
    volume[:, center_y, center_x] = 1.0  # Already set above

    return volume
