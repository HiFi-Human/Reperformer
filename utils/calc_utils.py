import torch
import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_inverse(q):
    if isinstance(q, torch.Tensor):
        conj_q = q.clone()
    elif isinstance(q, np.ndarray):
        conj_q = q.copy()
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array.")
    
    conj_q[..., 1:] *= -1
    
    if isinstance(q, torch.Tensor):
        return conj_q / q.norm(dim=-1, keepdim=True)
    elif isinstance(q, np.ndarray):
        norm = np.sqrt(np.sum(q**2, axis=-1, keepdims=True))
        return conj_q / norm

def norm_quaternion(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    return q

def build_rotation(q):
    r, x, y, z = q.unbind(-1)
    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    twx = tx * r
    twy = ty * r
    twz = tz * r
    txx = tx * x
    tyy = ty * y
    tzz = tz * z
    txy = tx * y
    txz = tx * z
    tyz = ty * z

    matrix = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - (tyy + tzz)
    matrix[..., 0, 1] = txy - twz
    matrix[..., 0, 2] = txz + twy
    matrix[..., 1, 0] = txy + twz
    matrix[..., 1, 1] = 1 - (txx + tzz)
    matrix[..., 1, 2] = tyz - twx
    matrix[..., 2, 0] = txz - twy
    matrix[..., 2, 1] = tyz + twx
    matrix[..., 2, 2] = 1 - (txx + tyy)
    return matrix

def norm_quaternion(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    return q

def batch_qvec2rotmat(qvecs):
    qvecs = np.array(qvecs)
    q0, q1, q2, q3 = qvecs[:, 0], qvecs[:, 1], qvecs[:, 2], qvecs[:, 3]
    
    R = np.empty((qvecs.shape[0], 3, 3))

    R[:, 0, 0] = 1 - 2 * q2**2 - 2 * q3**2
    R[:, 0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[:, 0, 2] = 2 * q1 * q3 + 2 * q0 * q2

    R[:, 1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[:, 1, 1] = 1 - 2 * q1**2 - 2 * q3**2
    R[:, 1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    R[:, 2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[:, 2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[:, 2, 2] = 1 - 2 * q1**2 - 2 * q2**2

    return R

def batch_qvec2rotmat_torch(qvecs):
    q0, q1, q2, q3 = qvecs[:, 0], qvecs[:, 1], qvecs[:, 2], qvecs[:, 3]
    
    R = torch.empty(qvecs.shape[0], 3, 3, device=qvecs.device)

    R[:, 0, 0] = 1 - 2 * q2**2 - 2 * q3**2
    R[:, 0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[:, 0, 2] = 2 * q1 * q3 + 2 * q0 * q2

    R[:, 1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[:, 1, 1] = 1 - 2 * q1**2 - 2 * q3**2
    R[:, 1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    R[:, 2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[:, 2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[:, 2, 2] = 1 - 2 * q1**2 - 2 * q2**2

    return R

def batch_qvec2rotmat_torch2(qvecs):
    q0, q1, q2, q3 = qvecs[:, :, 0], qvecs[:, :, 1], qvecs[:, :, 2], qvecs[:, :, 3]
    
    R = torch.empty(qvecs.shape[0], qvecs.shape[1], 3, 3, device=qvecs.device)

    R[:, :, 0, 0] = 1 - 2 * q2**2 - 2 * q3**2
    R[:, :, 0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[:, :, 0, 2] = 2 * q1 * q3 + 2 * q0 * q2

    R[:, :, 1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[:, :, 1, 1] = 1 - 2 * q1**2 - 2 * q3**2
    R[:, :, 1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    R[:, :, 2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[:, :, 2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[:, :, 2, 2] = 1 - 2 * q1**2 - 2 * q2**2

    return R

def batch_rotmat2qvec(Rs):
    Rs = np.array(Rs)
    qvecs = np.empty((Rs.shape[0], 4))

    for i, R in enumerate(Rs):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        qvecs[i] = qvec

    return qvecs

def batch_rotmat2qvec_torch(Rs):
    epsilon = 1e-6  
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2], min=epsilon)).unsqueeze(1)
    qx = (Rs[:, 2, 1] - Rs[:, 1, 2]).unsqueeze(1) / (4.0 * qw + epsilon)
    qy = (Rs[:, 0, 2] - Rs[:, 2, 0]).unsqueeze(1) / (4.0 * qw + epsilon)
    qz = (Rs[:, 1, 0] - Rs[:, 0, 1]).unsqueeze(1) / (4.0 * qw + epsilon)
    qvecs = torch.cat([qw, qx, qy, qz], dim=1)

    qvecs[qvecs[:, 0] < 0] *= -1

    return qvecs