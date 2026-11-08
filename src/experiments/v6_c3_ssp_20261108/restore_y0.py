"""还原初态：从 test_y0 仿射逆变换还原初态。
假设参数：
C = [-0.9009469748, -0.4014121294]
A = [0.876341, 0.876]
"""

import numpy as np

C_AFFINE = np.array([-0.9009469748, -0.4014121294], dtype=np.float64)
A_AFFINE = np.array([0.876341, 0.876], dtype=np.float64)

def restore_initial_y0(test_y0_raw, clip_height=True, clip_agb=True):
    """Restores unscaled physical initial height and AGB from raw test_y0.
    Args:
        test_y0_raw: Array of shape (..., >=2) containing transformed initial values.
        clip_height: Whether to clip restored height to [0, 35].
        clip_agb: Whether to clip restored AGB to >= 0.
    Returns:
        Array of shape (..., 2) with restored physical values.
    """
    raw = np.asarray(test_y0_raw, dtype=np.float64)
    y0 = (raw[..., :2] - C_AFFINE) / A_AFFINE
    if clip_height:
        y0[..., 0] = np.clip(y0[..., 0], 0.0, 35.0)
    if clip_agb:
        y0[..., 1] = np.maximum(y0[..., 1], 0.0)
    return y0
