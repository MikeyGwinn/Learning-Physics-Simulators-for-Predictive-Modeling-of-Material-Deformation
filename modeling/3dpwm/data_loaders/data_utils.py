import numpy as np
import torch

# converted from velocity to displacement
def calculate_displacements(list_of_data):
    """
    Computes displacements for each point cloud in the list.
    The point clouds are from earliers to latest.
    Args:
        list_of_data: list of data starting from the oldest
    Returns:
        output_list_of_data: list of data with displacements
    """

    positions_over_T = [None] * len(list_of_data)

    # Extract positions
    for i, data in enumerate(list_of_data):
        positions_over_T[i] = data["positions"]

    positions = np.stack(positions_over_T)
    displacements = np.zeros_like(positions)
    # leave the first displacement as 0s and start from index 1.
    displacements[1:] = positions[1:] - positions[:-1]

    for i, data in enumerate(list_of_data):
        list_of_data[i]["displacements"] = displacements[i]

    return list_of_data


def ptv3_move_to_gpu(point_dict, device):
    """
    Move a point dictionary to a GPU.
    Args:
        point_dict: A dictionary containing the point data.
        device: The device to move the data to.
    Returns:
        point_dict: The point dictionary with the data moved to the GPU.
    """
    if isinstance(point_dict, list):
        for point_idx in range(len(point_dict)):
            for key in point_dict[point_idx].keys():
                if isinstance(point_dict[point_idx][key], torch.Tensor):
                    point_dict[point_idx][key] = point_dict[point_idx][key].to(device)
                else:
                    point_dict[point_idx][key] = torch.tensor(
                        point_dict[point_idx][key]
                    ).to(device)
    else:
        for key in point_dict.keys():
            if isinstance(point_dict[key], torch.Tensor):
                point_dict[key] = point_dict[key].to(device)
            else:
                point_dict[key] = torch.tensor(point_dict[key]).to(device)

    return point_dict
