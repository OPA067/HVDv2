from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def np_softmax(X, theta=1.0, axis=0):
    y = np.atleast_2d(X * 100)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    r50 = metrics['R50']
    mr = metrics['MR']
    meanr = metrics["MeanR"]
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - R@50: {:.4f} - Median R: {} - MeanR: {}'.format(r1, r5, r10, r50,
                                                                                          mr, meanr))


# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k=[1, 5, 10, 50]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    # assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results


def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T


if __name__ == '__main__':
    test_sim = np.random.rand(1000, 1000)
    metrics = compute_metrics(test_sim)
    print_computed_metrics(metrics)
