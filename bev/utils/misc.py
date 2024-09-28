from typing import Mapping, Sequence

import torch
from torch import Tensor


def inverse_trans_matrix(matrix: Tensor):
    """Inverse transformation matrix(es)

    """
    assert matrix.shape[-1] == 4
    assert matrix.shape[-2] == 4
    batch_dims = matrix.shape[:-2]
    matrix = matrix.reshape(-1, 4, 4)
    batch = matrix.shape[0]

    if isinstance(matrix, Tensor):
        device = matrix.device
        rot = matrix[..., :3, :3]
        trans = matrix[..., :3, 3:4]

        if torch.allclose(rot @ rot.mT,
                          torch.eye(3, device=device).repeat(batch, 1, 1),
                          atol=1e-5):
            rot_inv = rot.mT
        else:
            rot_inv = torch.inverse(rot)
        trans_inv = -rot_inv @ trans

        matrix_inv = torch.eye(4, device=device).repeat(*batch_dims, 1, 1)
        matrix_inv[..., :3, :3] = rot_inv
        matrix_inv[..., :3, 3:4] = trans_inv
    else:
        raise NotImplementedError
    matrix_inv = matrix_inv.reshape(*batch_dims, 4, 4)

    return matrix_inv


def cast_data(data, data_type):
    if isinstance(data, Mapping):
        return {key: cast_data(data[key], data_type) for key in data}
    elif isinstance(data, (str, bytes)) or data is None:
        return data
    elif isinstance(data, Sequence):
        return type(data)(cast_data(sample, data_type) for sample in data)  # type: ignore  # noqa: E501
    elif isinstance(data, torch.Tensor):
        return data.to(data_type)
    else:
        return data



