from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import copy
import torch
import logging
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch import nn
from torchmetrics import Metric
from tqdm import tqdm
from omegaconf import open_dict

from ml_utilities.output_loader.result_loader import JobResult
from ml_utilities.utils import get_device, hyp_param_cfg_to_str
from ml_utilities.run_utils.run_handler import EXP_NAME_DIVIDER

from ml_utilities.data.datasetgenerator import DatasetGenerator

LOGGER = logging.getLogger(__name__)


def interpolate_linear_runs(
        run_0: JobResult,
        run_1: JobResult,
        score_fn: Union[nn.Module, Metric],
        model_idx: Union[int, List[int]] = [-1, -2],
        interpolation_factors: torch.Tensor = torch.linspace(0.0, 1.0, 5),
        interpolate_linear_kwargs: Dict[str, Any] = {},
        device: Union[torch.device, str, int] = 'auto',
        return_dataframe: bool = True,
        dataset_generator: DatasetGenerator = None
) -> Union[Dict[str, Any], Tuple[pd.DataFrame, Optional[pd.DataFrame]]]: # TODO pass dataloader here already
    """Interpolate linearly between models of two runs. 

    Args:
        run_0 (JobResult): Run 0.
        run_1 (JobResult): Run 1.
        score_fn (Union[nn.Module, Metric]): Performance measure for the models.
        model_idx (Union[int, List[int]], optional): The model index/indices used for linear interpolation. Defaults to -1.
                                                     If -1, use the respective best model.
                                                     If -2, use the respective model with the highest model index or the model 
                                                     with the longest training duration.
        interpolation_factors (torch.Tensor, optional): Interpolation factors. Defaults to torch.linspace(0.0, 1.0, 5).
        interpolate_linear_kwargs (Dict[str, Any], optional): Some further keyword arguments for `interpolate_linear`. Defaults to {}.
        device (Union[torch.device, str, int], optional): Device for linear interpolation. Defaults to 'auto'.
        return_dataframe (bool, optional): If true, return results as dataframes. Return a dictionary otherwise. Defaults to True.
        dataset_generator (dataset_generator, optional): Use this dataset generator to access dataset. 
                                                         For efficiency only, if this method is called multiple times. 
                                                         If None, create dataset_generator from run_0's data config. Defaults to None.

    Raises:
        ValueError: If a model index is missing in one of the two runs.

    Returns:
        Union[Dict[str, Any], Tuple[pd.DataFrame, Optional[pd.DataFrame]]]: Results as dataframes or a single dictionary.
    """
    device = get_device(device)
    if isinstance(model_idx, int):
        model_idx = [model_idx]

    # use run_0 to determine interpolation name and seeds
    interpolation_name = run_0.experiment_name + EXP_NAME_DIVIDER + hyp_param_cfg_to_str(run_0.override_hpparams)
    interpolation_seeds = (run_0.experiment_data.seed, run_1.experiment_data.seed)

    if dataset_generator is None:
        # use dataset from run_0 for dataset setup
        data_cfg = copy.deepcopy(run_0.config.config.data)
        # set training augmentations to the validation augmentations
        with open_dict(data_cfg):
            data_cfg.train_split_transforms = data_cfg.get('val_split_transforms', {})

        ds_generator = DatasetGenerator(**data_cfg)
        ds_generator.generate_dataset()
    else:
        ds_generator = dataset_generator
        if not ds_generator.dataset_generated:
            ds_generator.generate_dataset()

    other_datasets = {'val': ds_generator.val_split}
    if 'other_datasets' in interpolate_linear_kwargs:
        other_datasets.update(interpolate_linear_kwargs['other_datasets'])

    res_dict = {}  # contains all interpolation results as dictionary
    dataset_series: List[pd.Series] = []
    distance_series: List[pd.Series] = []

    runs = [run_0, run_1]
    for midx in model_idx:
        models = []
        m_idxes = []
        for i, r in enumerate(runs):
            if midx == -1:
                run_midx = r.best_model_idx
            elif midx == -2:
                run_midx = r.highest_model_idx
            else:
                run_midx = midx
            
            m_idxes.append(run_midx)
            try:
                m = r.get_model_idx(idx=run_midx, device=device)
            except FileNotFoundError:
                raise ValueError(f'Missing model_idx={run_midx} in run_{i}: {r}')
            models.append(m)

        model_0, model_1 = models

        idx_res_dict = interpolate_linear(model_0=model_0,
                                          model_1=model_1,
                                          score_fn=score_fn,
                                          train_dataset=ds_generator.train_split,
                                          interpolation_factors=interpolation_factors,
                                          other_datasets=other_datasets,
                                          **interpolate_linear_kwargs)
        res_dict[tuple(m_idxes)] = idx_res_dict
        # convert result dict into more readable dataframe
        idx_dataset_series, idx_distance_series = interpolation_result2series(idx_res_dict)
        dataset_series.append(idx_dataset_series)
        distance_series.append(idx_distance_series)

    ret_val = res_dict
    # create dataframes
    if return_dataframe:
        ind = pd.MultiIndex.from_product([[interpolation_name], [interpolation_seeds],
                                          list(res_dict.keys())],
                                         names=['job', 'seeds', 'model_idxes'])
        datasets_df = pd.DataFrame(dataset_series, index=ind)
        distances_df = None
        if not distance_series[0] is None:
            distances_df = pd.DataFrame(distance_series, index=ind)
        ret_val = datasets_df, distances_df
    return ret_val
    # return res_dict, dataset_series, distance_series


def interpolation_result2series(result_dict: Dict[str, Any]) -> Tuple[pd.Series, Optional[pd.Series]]:
    ds_key = 'datasets'
    dist_key = 'distances'

    def ds_result2series(ds_result_dict: Dict[str, Any], interp_factors: np.ndarray) -> pd.Series:
        instability_key = 'instability'
        interp_sc_key = 'interpolation_scores'
        interp_scores = np.array(ds_result_dict[interp_sc_key])
        assert len(interp_factors) == len(interp_scores)
        # create results dictionary with necessary values
        res_dict = {alpha: interp_score for alpha, interp_score in zip(interp_factors, interp_scores)}
        res_dict[instability_key] = ds_result_dict[instability_key]
        # create index
        interp_ind = np.full_like(interp_scores, interp_sc_key, dtype=object)
        idx_tuples = list(zip(interp_ind, interp_factors))
        idx_tuples.append((instability_key, None))
        ind = pd.MultiIndex.from_tuples(idx_tuples, names=['score', 'alpha'])
        return pd.Series(res_dict.values(), index=ind)

    interp_factors = np.array(result_dict['interpolation_factors'])
    ds_dict = result_dict[ds_key]
    ds_series_dict = {ds_name: ds_result2series(ds_result, interp_factors) for ds_name, ds_result in ds_dict.items()}
    dataset_series = pd.concat(ds_series_dict, names=[ds_key])

    distances_series = None
    if dist_key in result_dict:
        distances_dict = result_dict[dist_key]
        ind = pd.MultiIndex.from_arrays([distances_dict.keys()], names=[dist_key])
        distances_series = pd.Series(distances_dict.values(), index=ind)

    return dataset_series, distances_series


def interpolate_linear(model_0: nn.Module,
                       model_1: nn.Module,
                       train_dataset: data.Dataset,
                       score_fn: Union[nn.Module, Metric],
                       other_datasets: Dict[str, data.Dataset] = {}, # TODO pass
                       interpolation_factors: torch.Tensor = torch.linspace(0.0, 1.0, 5),
                       dataloader_kwargs: Dict[str, Any] = {
                           'batch_size': 256,
                           'drop_last': False,
                           'num_workers': 4,
                           'persistent_workers': True,
                           'pin_memory': True
                       },
                       compute_model_distances: bool = True,
                       interpolation_on_train_data: bool = True,
                       update_bn_statistics: bool = True,
                       tqdm_desc: str = 'Alphas') -> Dict[str, Any]:
    """Interpolate linearly between two models. Evaluates the performance of each interpolated model on given datasets.
    
    Note:
        Also computes the instability value according to Frankle et al., 2020, p. 3.
        Instability = max/min [interpolation_scores] - mean[interpolation_score(0.0), interpolation_score(1.0)]
    
    References:
        Frankle, Jonathan, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. 2020. 
            “Linear Mode Connectivity and the Lottery Ticket Hypothesis.” arXiv. http://arxiv.org/abs/1912.05671.

    Args:
        model_0 (nn.Module): First model.
        model_1 (nn.Module): Second model.
        train_dataset (data.Dataset): Dataset on which the models have been trained. 
                                      If applicable, this dataset is used to recompte batch norm statistics.
        score_fn (Union[nn.Module, Metric]): The performance measure on which each model is used.
        other_datasets (Dict[str, data.Dataset], optional): Evaluation dataset with descriptor as key. Defaults to {}.
        interpolation_factors (torch.Tensor, optional): Interpolation factor for linear interpolation. Defaults to torch.linspace(0.0, 1.0, 5).
        dataloader_kwargs (Dict[str, Any], optional): Additional dataloader keyword arguments. Defaults to {'batch_size': 256}.
        compute_model_distances (bool, optional): Computes distance metrics on given models. Defaults to True.
        interpolation_on_train_data (bool, optional): Evaluates interpolation performance on train data too. Defaults to False.
        update_bn_statistics (bool, optional): Update batchnorm statistics before evaluation.
        tqdm_desc (str, optional): The description for the tqdm progress bar. If '', then no progress bar is displayed. Defaults to 'Alphas'.

    Raises:
        ValueError: If no eval datasets are given or the model architectures do not match.

    Returns:
        Dict[str, Any]: Dictionary containing the results.
                        Example:
                           {'datasets': {'val': {'instability': -0.1300981044769287,
                                                 'interpolation_scores': [0.974958598613739,
                                                                          0.9691435694694519,
                                                                          0.976451575756073]}},
                            'distances': {'cosinesimilarity': 0.010202181525528431,
                                          'l2distance': 30.355226516723633},
                            'interpolation_factors': [0.0, 0.25, 0.5, 0.75, 1.0]}
    """
    get_model_device = lambda model: next(iter(model.parameters())).device
    assert get_model_device(model_0) == get_model_device(model_1), f'Models to interpolate not on same device!'
    device = get_model_device(model_0)
    assert 'train' not in other_datasets, f'`train` is a reserved dataset name. Please rename this evaluation dataset.'
    assert interpolation_factors.dim() == 1, '`interpolation_factors` must be tensor of dimension 1.'
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def eval_loop(model: nn.Module, dataloader: data.DataLoader, score_fn: nn.Module) -> float:
        batch_scores = []
        for batch_idx, (xs, ys) in enumerate(dataloader):
            xs, ys = xs.to(device), ys.to(device)
            with torch.no_grad():
                y_pred = model(xs)
                score = score_fn(y_pred, ys)
                batch_scores.append(score)
        return torch.tensor(batch_scores).mean().item()

    # check if models have batch_norm layers
    models_have_batch_norm_layers = False
    for m in model_0.modules():
        if isinstance(m, bn_types):
            models_have_batch_norm_layers = True
            break

    # prepare datasets and results dict
    eval_datasets = copy.copy(other_datasets)  # shallow copy
    if interpolation_on_train_data:
        eval_datasets['train'] = train_dataset  # reference only
    ds_dict = {ds_name: [] for ds_name in eval_datasets}

    # add base_interpolation_factors (=interpolation factors that correspond to base models)
    # if not already in the interpolation factors
    base_interpolation_factors = [0.0, 1.0]
    interpolation_factors = interpolation_factors.tolist()
    for f in base_interpolation_factors:
        if not f in interpolation_factors:
            interpolation_factors.append(f)
    interpolation_factors.sort()
    interpolation_factors = torch.tensor(interpolation_factors)

    res_dict = {}
    res_dict['interpolation_factors'] = interpolation_factors.tolist()

    # create eval_dataloaders
    eval_dataloaders = {ds_name: data.DataLoader(ds, **dataloader_kwargs) for ds_name, ds in eval_datasets.items()}
    if not eval_dataloaders:
        raise ValueError(
            'No evaluation datasets provided. Pass eval_datasets or set `interpolation_on_train_data=True`.')

    interpolation_factors = interpolation_factors.to(device)
    score_fn = score_fn.to(device)
    if tqdm_desc:
        it = tqdm(interpolation_factors, desc=tqdm_desc, file=sys.stdout)
    else:
        it = interpolation_factors
    # alpha = interpolation factor
    for alpha in it:
        # create interpolated model in a memory friendly way (only use memory used for another model instance)
        interp_model = copy.deepcopy(model_0)
        interp_model_state_dict = interp_model.state_dict()
        for (k0, v0), (k1, v1) in zip(model_0.state_dict().items(), model_1.state_dict().items()):
            if k0 != k1:
                raise ValueError(f'Model architectures do not match: {k0} != {k1}')
            # skip parameters that are non-float, e.g. 'num_batches_tracked' of batchnorm layers
            if v0.is_floating_point():
                torch.lerp(v0, v1, alpha, out=interp_model_state_dict[k0])  # linear interpolation between weights
        interp_model.load_state_dict(interp_model_state_dict)

        if models_have_batch_norm_layers and update_bn_statistics:
            # reset running stats
            # compute batch_norm statistics on train_dataset
            # rely on torch.optim.swa_utils.uptade_bn, since it takes care of batchnorm momentum
            train_loader = eval_dataloaders.get('train', None)
            if train_loader is None:
                train_loader = data.DataLoader(train_dataset, **dataloader_kwargs)
            torch.optim.swa_utils.update_bn(loader=train_loader, model=interp_model, device=device)

        interp_model.train(False)
        # eval on eval_datasets
        for ds_name, dataloader in eval_dataloaders.items():
            score = eval_loop(model=interp_model, dataloader=dataloader, score_fn=score_fn)
            ds_dict[ds_name].append(score)

    if compute_model_distances:
        vec_0 = nn.utils.parameters_to_vector(model_0.parameters())
        vec_1 = nn.utils.parameters_to_vector(model_1.parameters())
        distances = {}
        # L2 distance
        distances['l2distance'] = torch.linalg.norm(vec_1 - vec_0).item()
        # cosine similarity
        distances['cosinesimilarity'] = nn.functional.cosine_similarity(vec_0, vec_1, dim=0).item()
        res_dict['distances'] = distances

    # compute instability value
    # find weight indices for base models
    # (necessary if 0. and 1. value not at beginning or end of interpolation_factors tensor)
    original_models_in_interpolation_factors = True
    interp_factors_list = interpolation_factors.tolist()
    base_interpolation_factor_idxes = []
    for f in base_interpolation_factors:
        try:
            base_interpolation_factor_idxes.append(interp_factors_list.index(f))
        except ValueError:
            original_models_in_interpolation_factors = False
            LOGGER.warning(f'Interpolation factor {f} missing. Cannot compute instability value!')
            break

    # compute instability per dataset
    def compute_instability(interp_scores: List[float]) -> float:
        if original_models_in_interpolation_factors:
            interp_scores = torch.tensor(interp_scores)
            base_mean = interp_scores[base_interpolation_factor_idxes].mean()
            torch_minmax = torch.min if score_fn.higher_is_better else torch.max
            interp_scores_between_bases = interp_scores[base_interpolation_factor_idxes[0]:(
                base_interpolation_factor_idxes[1] + 1)]
            instability = torch_minmax(interp_scores_between_bases) - base_mean
            return instability.item()
        else:
            return float('nan')

    for k, v in ds_dict.items():
        # k is ds_name, v is interpolation_scores
        ds_dict[k] = {'instability': compute_instability(v), 'interpolation_scores': v}

    res_dict['datasets'] = ds_dict

    return res_dict