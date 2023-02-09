import copy
from typing import Any, Dict, List, Tuple, Union
import torch
import logging
import pandas as pd
from torch import nn
from torch.utils import data
from tqdm import tqdm
from pathlib import Path

from ml_utilities.torch_utils import gradients_to_vector, get_loss
from ml_utilities.output_loader import JobResult
from ml_utilities.utils import get_device
from ml_utilities.data.datasetgenerator import DatasetGenerator
from ml_utilities.run_utils.runner import Runner
from ml_utilities.pandas_utils import save_df_dict, load_df_dict_pickle

LOGGER = logging.getLogger(__name__)
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


def compute_covariance_gram_spectrum_statistics(input_matrix: torch.Tensor) -> Tuple[Dict[str, float], List[float]]:
    """Computes the eigenvalues, maximum eigenvalue, minimum eigenvalue, inverse condition number, 
    and effective rank of the covariance gram matrix."""

    n_samples = input_matrix.shape[1]  # L observations
    cov_gram_matrix = compute_covariance_gram_matrix(input_matrix)  # symmetric, positive definite

    # compute eigenvalues, remember for SPD (symmetric, positive definite) matrices the eigendecomposition is equal to the SVD
    eigvals = torch.linalg.svdvals(cov_gram_matrix)  # these are sorted in descending order
    # eigenspectrum
    max_eigval = eigvals[0]
    # smallest non-zero eigenvalue
    min_non_zero_eigval = eigvals[
        n_samples -
        2]  # the last eigenvalue is zero (covariance matrix computed using L observations has by definition L − 1 non-zero eigenvalues)
    inverse_condition_number = min_non_zero_eigval / max_eigval
    # effective rank of the covariance gram matrix
    erank = torch.exp(torch.distributions.Categorical(probs=eigvals).entropy())
    ret_dict = {
        'max_eigval': max_eigval.item(),
        'min_non_zero_eigval': min_non_zero_eigval.item(),
        'inverse_condition_number': inverse_condition_number.item(),
        'erank': erank.item()
    }
    return ret_dict, eigvals.tolist()


def gradient_covariance_analysis(model: nn.Module,
                                 loss_fn: nn.Module,
                                 dataloaders: Dict[str, data.DataLoader],
                                 num_batches: int = 25,
                                 device: torch.device = None,
                                 use_tqdm: bool = True) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
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
        Dict[str, Dict[str, float]]: A dictionary of dictionaries. 
                                     The keys are the names of the dataloaders and the values are dictionaries of the results.
    """
    if device is None:
        device = next(iter(model.parameters())).device
    else:
        model.to(device)

    model.eval()
    model.zero_grad()

    covariance_stats_per_dataloader = {}
    eigvals_per_dataloader = {}
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
        covariance_stats_per_dataloader[name], eigvals_per_dataloader[
            name] = compute_covariance_gram_spectrum_statistics(gradient_matrix)

    return covariance_stats_per_dataloader, eigvals_per_dataloader


class GradCovarianceAnalyzer(Runner):

    """Performs a gradient covariance analysis on checkpoints of a single job."""

    save_folder_basename = 'grad_covariance_analysis'
    save_folder_combined_results = 'combined_results'
    save_folder_single_results = 'single_results'

    save_filename_param_suffix = '--nb-{num_batches}--bs-{batch_size}'
    save_filename = 'covariance_analysis_results'

    def __init__(
        self,
        job: Union[Path, JobResult],
        num_batches: Union[int, List[int]] = 25,
        batch_sizes: Union[int, List[int]] = 128,
        checkpoint_idxs: Union[int, List[int]] = [],  # if empty, all available checkpoints are used
        device: Union[torch.device, str, int] = 'auto',
        other_datasets: Dict[str, data.Dataset] = {},
        dataloaders: Dict[str, data.DataLoader] = {},
        dataloader_kwargs: Dict[str, Any] = {
            'drop_last': False,
            'num_workers': 4,
            'persistent_workers': True,
            'pin_memory': True
        },
        save_folder_suffix: str = '',
        save_to_disk: bool = True,
    ):
        super().__init__()
        if isinstance(job, Path):
            job = JobResult(job)
        self.job = job

        if not isinstance(checkpoint_idxs, list):
            self.checkpoint_idxs = [checkpoint_idxs]
        elif len(checkpoint_idxs) == 0:
            self.checkpoint_idxs = self.job.available_model_checkpoint_indices
        else:
            self.checkpoint_idxs = checkpoint_idxs

        self.device = get_device(device)

        self.num_batches = num_batches if isinstance(num_batches, list) else [num_batches]
        self.batch_sizes = batch_sizes if isinstance(batch_sizes, list) else [batch_sizes]

        self.other_datasets = other_datasets
        self.dataloader_kwargs = dataloader_kwargs

        self.user_dataloaders = dataloaders
        if self.user_dataloaders:
            if len(self.batch_sizes) > 1:
                raise ValueError('If dataloaders are provided, only a single batch size can be used.')
            else:
                assert next(iter(self.user_dataloaders.values())).batch_size == self.batch_sizes[0]
        else:
            self.datasets = self._create_datasets()

        #* create loss
        loss_cls = get_loss(job.config.config.trainer.loss)
        self.loss_fn = loss_cls(reduction='mean')

        #* create save folders
        self.save_folder_suffix = save_folder_suffix
        self.save_to_disk = save_to_disk
        self.reload_results = False
        if self.save_to_disk:
            try:
                self.runner_dir, self.combined_results_dir, self.single_results_dir = self._create_save_folders(
                    save_folder_suffix)
            except FileExistsError:
                self.reload_results = True
                LOGGER.warning(
                    'The save folder already exists. Reloading those results. To recompute choose different save_folder_suffix.'
                )

    def _create_datasets(self) -> Dict[str, data.Dataset]:
        # use same transforms on train split as on val split
        data_cfg = copy.deepcopy(self.job.config.config.data)
        val_split_transforms = data_cfg.get('val_split_transforms', {})
        data_cfg.train_split_transforms = val_split_transforms
        self.dataset_gen = DatasetGenerator(**data_cfg)
        self.dataset_gen.generate_dataset()
        datasets = {'train': self.dataset_gen.train_split, 'val': self.dataset_gen.val_split}
        datasets.update(self.other_datasets)

        return datasets

    def _create_dataloaders(self, batch_size: int) -> Dict[str, data.DataLoader]:
        # use shuffle=False, this ensures that on every loop over the dataloader the same samples are used
        dataloaders = {
            name: data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)
            for name, dataset in self.datasets.items()
        }
        return dataloaders

    def _create_save_folders(self, save_folder_suffix: str = '') -> Tuple[Path, Path]:
        if save_folder_suffix:
            save_folder_name = f'{self.save_folder_basename}--{save_folder_suffix}'
        else:
            save_folder_name = self.save_folder_basename

        runner_dir = self.job.directory / save_folder_name
        combined_results_dir = self.job.directory / save_folder_name / self.save_folder_combined_results
        single_results_dir = self.job.directory / save_folder_name / self.save_folder_single_results

        combined_results_dir.mkdir(parents=True, exist_ok=False)
        single_results_dir.mkdir(parents=True, exist_ok=False)
        return runner_dir, combined_results_dir, single_results_dir

    def _covariance_analysis_for_batchsize_and_numbatches(self, batch_size: int, dataloaders: List[data.DataLoader],
                                                          num_batch: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        covariance_stats_results = {}
        covariance_eigvals_results = {}

        #* gradient analysis at checkpoints
        iter = tqdm(self.checkpoint_idxs)
        for cp_idx in iter:
            iter.set_description_str(f'cov analysis at checkpoint {cp_idx}')
            model = self.job.get_model_idx(cp_idx)
            cov_stats, cov_eigvals = gradient_covariance_analysis(model=model,
                                                                  loss_fn=self.loss_fn,
                                                                  dataloaders=dataloaders,
                                                                  num_batches=num_batch,
                                                                  device=self.device,
                                                                  use_tqdm=False)

            cov_stats_reform = {(ds_split, statistic_name): value for ds_split, subdict in cov_stats.items()
                                for statistic_name, value in subdict.items()}
            covariance_stats_results[cp_idx] = pd.Series(cov_stats_reform)
            covariance_eigvals_results[cp_idx] = pd.DataFrame(cov_eigvals).transpose()

        # prepend num_batches and batch_size to columns
        cov_stats_df = pd.DataFrame(covariance_stats_results).transpose()
        cov_stats_df.index = pd.MultiIndex.from_product([[batch_size], [num_batch], cov_stats_df.index],
                                                        names=('batch_size', 'num_batches', 'checkpoint_idx'))
        cov_stats_df.columns.names = ['dataset', 'spectral_statistic']

        cov_eigvals_df = pd.concat(covariance_eigvals_results)
        cov_eigvals_df.index.names = ['checkpoint_idx', 'dataset']
        cov_eigvals_df.columns.names = ['eigval_idx']
        cov_eigvals_df.index = pd.MultiIndex.from_product(
            [[batch_size], [num_batch],
             cov_eigvals_df.index.get_level_values('checkpoint_idx').unique(),
             cov_eigvals_df.index.get_level_values('dataset').unique()],
            names=('batch_size', 'num_batches', 'checkpoint_idx', 'dataset'))

        #* save results
        if self.save_to_disk:
            df_dict = {'cov_stats': cov_stats_df, 'cov_eigvals': cov_eigvals_df}
            fn_wo_ending = self.save_filename + self.save_filename_param_suffix.format(batch_size=batch_size,
                                                                                       num_batches=num_batch)
            save_df_dict(df_dict, self.single_results_dir, fn_wo_ending)

        # free GPU memory
        del model
        torch.cuda.empty_cache()

        return cov_stats_df, cov_eigvals_df

    def covariance_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if self.reload_results:
            return GradCovarianceAnalyzer.reload(self.job, self.save_folder_suffix)

        combined_cov_stats = []
        combined_cov_eigvals = []

        for batch_size in self.batch_sizes:
            #* create datasets
            if self.user_dataloaders:
                dataloaders = self.user_dataloaders
            else:
                #* create dataloaders
                dataloaders = self._create_dataloaders(batch_size=batch_size)

            for num_batch in self.num_batches:
                LOGGER.info(f'covariance analysis for batch_size={batch_size}, num_batches={num_batch}:')
                cov_stats_df, cov_eigvals_df = self._covariance_analysis_for_batchsize_and_numbatches(
                    batch_size, dataloaders, num_batch)

                combined_cov_stats.append(cov_stats_df)
                combined_cov_eigvals.append(cov_eigvals_df)

        combined_cov_stats_df = pd.concat(combined_cov_stats)
        combined_cov_eigvals_df = pd.concat(combined_cov_eigvals)

        #* save results
        if self.save_to_disk:
            df_dict = {'cov_stats': combined_cov_stats_df, 'cov_eigvals': combined_cov_eigvals_df}
            save_df_dict(df_dict, self.combined_results_dir, self.save_filename)

        return combined_cov_stats_df, combined_cov_eigvals_df

    @staticmethod
    def reload(job: Union[Path, JobResult], save_folder_suffix: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(job, Path):
            job = JobResult(job)

        if save_folder_suffix:
            save_folder_name = f'{GradCovarianceAnalyzer.save_folder_basename}--{save_folder_suffix}'
        else:
            save_folder_name = GradCovarianceAnalyzer.save_folder_basename

        df_file_dir = job.directory / save_folder_name / GradCovarianceAnalyzer.save_folder_combined_results

        df_dict = load_df_dict_pickle(dir=df_file_dir, filename_wo_ending=GradCovarianceAnalyzer.save_filename)

        return df_dict['cov_stats'], df_dict['cov_eigvals']

    def run(self):
        self.covariance_analysis()


# TODO implement class for covariance analysis on sweep
# saves the results in the sweep directory and the individual results in the job directories
# make this class runnable via config