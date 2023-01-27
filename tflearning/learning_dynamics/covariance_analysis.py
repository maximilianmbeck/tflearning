import copy
from typing import Any, Dict, List, Tuple, Union
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from tqdm import tqdm
from pathlib import Path

from ml_utilities.torch_utils import gradients_to_vector, get_loss
from ml_utilities.output_loader import JobResult
from ml_utilities.utils import get_device
from ml_utilities.data.datasetgenerator import DatasetGenerator
"""Module for doing analyses of the covariance matrix of the gradients of a neural network."""


def compute_covariance_gram_matrix(input_matrix: torch.Tensor, correction: int = 1) -> torch.Tensor:
    """Computes an approximation of the covariance matrix of the input matrix using the Gram matrix.

    For details see Appendix C in [#1].

    [#1] Jastrzebski, Stanislaw, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho*, and Krzysztof Geras*. 2022. 
         “The Break-Even Point on Optimization Trajectories of Deep Neural Networks.” In . https://openreview.net/forum?id=r1g87C4KwB.

    Args:
        input_matrix (torch.Tensor): Rows are variables and columns are observations. (Typically the gradients of a neural network are in the columns.)
        correction (int, optional): Difference between the sample size and sample degrees of freedom. 
                                    Defaults to Bessel’s correction, correction = 1 which returns the unbiased estimate.
                                    Correction = 0 will return the simple average. Defaults to 1.

    Returns:
        torch.Tensor: The covariance gram matrix. 
                      If L is the number of observations (columns in `input_matrix`), 
                      then the output is a L x L matrix.
    """
    N_samples = input_matrix.shape[1]
    N_samples = input_matrix.shape[1]
    mean = input_matrix.mean(dim=1, keepdim=False)
    gram_matrix = torch.zeros((N_samples, N_samples), device=input_matrix.device)
    for i in range(N_samples):
        for j in range(i + 1):
            gram_matrix[i, j] = gram_matrix[j, i] = torch.dot(input_matrix[:, i] - mean, input_matrix[:, j] - mean)
    return gram_matrix / (N_samples - correction)


def compute_covariance_gram_spectrum_statistics(input_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Computes the eigenvalues, maximum eigenvalue, minimum eigenvalue, inverse condition number, 
    and effective rank of the covariance gram matrix."""

    cov_gram_matrix = compute_covariance_gram_matrix(input_matrix)

    # compute eigenvalues, remember for SPD (symmetric, positive definite) matrices the eigendecomposition is equal to the SVD
    eigvals = torch.linalg.svdvals(cov_gram_matrix)
    # eigenspectrum
    max_eigval = eigvals.max()
    min_eigval = eigvals.min()
    inverse_condition_number = min_eigval / max_eigval
    # effective rank of the covariance gram matrix
    erank = torch.exp(torch.distributions.Categorical(probs=eigvals).entropy())
    ret_dict = {
        '_eigvals': eigvals,
        'max_eigval': max_eigval,
        'min_eigval': min_eigval,
        'inverse_condition_number': inverse_condition_number,
        'erank': erank
    }
    return ret_dict


def gradient_covariance_analysis(model: nn.Module,
                                 loss_fn: nn.Module,
                                 dataloaders: Dict[str, data.DataLoader],
                                 num_batches: int = 25,
                                 device: torch.device = None,
                                 use_tqdm: bool = True):
    """Computes the eigenvalues, maximum eigenvalue, minimum eigenvalue, inverse condition number, 
    and effective rank of the covariance gram matrix of the gradients of a neural network.

    Args:
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        dataloaders (Dict[str, DataLoader]): A dictionary of dataloaders. 
                                            The keys are the names of the dataloaders and the values are the dataloaders.
        num_batches (int, optional): The number of batches to use for the analysis. Defaults to 25.
        device (torch.device, optional): The device to use for the analysis. Defaults to None.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: A dictionary of dictionaries. 
                                            The keys are the names of the dataloaders and the values are dictionaries of the results.
    """
    if device is None:
        device = next(iter(model.parameters())).device
    else:
        model.to(device)

    model.eval()
    model.zero_grad()

    covariance_stats_per_dataloader = {}

    if use_tqdm:
        iter = tqdm(dataloaders.items())
    else:
        iter = dataloaders.items()

    for name, dataloader in iter:
        gradients = []
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()

            gradient_vec = gradients_to_vector(model.parameters())
            gradients.append(gradient_vec)
            if i >= num_batches - 1:
                break
        gradient_matrix = torch.stack(gradients, dim=1)
        covariance_stats_per_dataloader[name] = compute_covariance_gram_spectrum_statistics(gradient_matrix)

    return covariance_stats_per_dataloader


# TODO implement function for covariance analysis on single job
# saves the results in the job directory
# make this module callable from the command line to perform this job
# returns #1 dataframe with aggregated spectrum results per checkpoint idx
# returns #2 dataframe with full spectrum results per dataloader
def gradient_covariance_analysis_single_job(
    job: Union[Path, JobResult],
    num_batches: int = 25,
    batch_size: int = 128,
    checkpoint_idxs: Union[int, List[int]] = -1,
    device: Union[torch.device, str, int] = 'auto',
    other_datasets: Dict[str, data.Dataset] = {},
    dataloaders: Dict[str, data.DataLoader] = {},
    dataloader_kwargs: Dict[str, Any] = {
        'drop_last': False,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True
    }
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(job, Path):
        job = JobResult(job)

    if not isinstance(checkpoint_idxs, list):
        checkpoint_idxs = [checkpoint_idxs]

    device = get_device(device)

    #* create datasets
    if not dataloaders:
        # use same transforms on train split as on val split
        data_cfg = copy.deepcopy(job.config.config.data)
        val_split_transforms = data_cfg.get('val_split_transforms', {})
        data_cfg.train_split_transforms = val_split_transforms

        dataset_gen = DatasetGenerator(**data_cfg)
        dataset_gen.generate_dataset()

        datasets = {'train': dataset_gen.train_split, 'val': dataset_gen.val_split}
        datasets.update(other_datasets)

        #* create dataloaders
        # use shuffle=False, this ensures that on every loop over the dataloader the same samples are used
        dataloaders = {
            name: data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs) for name, dataset in datasets.items()
        }

    #* create loss
    loss_cls = get_loss(job.config.trainer.loss)
    loss_fn = loss_cls(reduction='mean')

    #* gradient analysis at checkpoints
    # TODO from here


# TODO implement class for covariance analysis on sweep
# saves the results in the sweep directory and the individual results in the job directories
# make this class runnable via config