import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
from ml_utilities.output_loader.result_loader import JobResult, SweepResult
from ml_utilities.torch_utils import compute_grad_norm
from ml_utilities.utils import get_device
from tflearning.data.creator import create_datasetgenerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# take as input a sweep summary
# epoch at which the el2n is computed

# use the config from the first run to load data etc.
"""
The algorithm is as follows:
- for every model in the sweep, compute the el2n, save the scores (make sure data is in the same order)
  - el2n is the l2 norm of the error vector (difference between the prediction and the ground truth one-hot vector)

  
Return a dictionary with the following structure:
{el2n: {single_runs: {job_name: scores}, average: scores }, grand: {single_runs: {job_name: scores}, average: scores }}
"""


@dataclass
class El2nConfig:
    sweep_result: Union[SweepResult, str]
    compute_at_progress_idx: int
    batch_size: int = 128
    compute_grand: bool = True
    compute_el2n: bool = True
    gpu_id: int = 0


class El2nAndGrandScores:

    def __init__(self,
                 sweep_result: Union[SweepResult, str],
                 compute_at_progress_idx: int,
                 compute_grand: bool = True,
                 compute_el2n: bool = True,
                 batch_size: int = 128,
                 gpu_id: int = 0,
                 dataset_generate_init: Callable = create_datasetgenerator):
        if isinstance(sweep_result, str):
            sweep_result = SweepResult(sweep_result)
        self.sweep_result = sweep_result
        self.compute_at_progress_idx = compute_at_progress_idx
        self.batch_size = batch_size
        self.device = get_device(gpu_id)

        failed_runs, _ = self.sweep_result.get_failed_jobs()
        assert len(failed_runs) == 0, f"Found {len(failed_runs)} failed runs: {failed_runs}"

        self.runs = self.sweep_result.get_jobs()

        self.data_cfg = self.runs[0].config.config.data
        self.dataset_generator = dataset_generate_init(self.data_cfg)
        self.dataset_generator.generate_dataset()
        train_dataset = self.dataset_generator.train_split

        self.el2n_dataloader = None
        if compute_el2n:
            self.el2n_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.grand_dataloader = None
        if compute_grand:
            self.grand_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.score_dictionary = {}

    def compute(self) -> Dict[str, Any]:
        el2n_run_scores = {}
        grand_run_scores = {}
        for run in tqdm(self.runs, file=sys.stdout, desc="Models"):
            if self.el2n_dataloader is not None:
                el2n_scores, labels = self._compute_el2n_for_run(run, self.el2n_dataloader, self.compute_at_progress_idx)
                el2n_run_scores.update(el2n_scores)
            if self.grand_dataloader is not None:
                grand_scores, labels = self._compute_grand_for_run(run, self.grand_dataloader)
                grand_run_scores.update(grand_scores)

        result_dict = {}
        if self.el2n_dataloader is not None:
            result_dict["el2n"] = self._create_result_dictionary(el2n_run_scores)
        if self.grand_dataloader is not None:
            result_dict["grand"] = self._create_result_dictionary(grand_run_scores)
        
        result_dict["labels"] = labels

        from datetime import datetime
        self._save(result_dict, self.sweep_result.directory / f"el2n_grand_scores_{datetime.now().strftime('%y%m%d_%H%M%S')}.p")

        # return el2n_run_scores, grand_run_scores
        return result_dict

    def _save(self, scores: Dict[str, Any], filename: str):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(scores, f)

    def _create_result_dictionary(self, run_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Creates a dictionary with the following structure:
        {single_runs: {job_name: scores}, average: scores }"""
        return {"single_runs": run_scores, "average": self._compute_average_scores(run_scores)}

    def _compute_average_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(list(scores.values())), axis=0)

    def _compute_el2n_for_run(self, run: JobResult, dataloader: DataLoader, progress_idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        model = run.get_model_idx(progress_idx).to(self.device)
        scores, labels = compute_el2n_for_model(model=model, dataloader=dataloader, device=self.device)
        return {run.directory.name: scores}, labels

    def _compute_grand_for_run(self,
                               run: JobResult,
                               dataloader: DataLoader,
                               progress_idx: int = 0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        model = run.get_model_idx(progress_idx).to(self.device)
        scores, labels = compute_grand_for_model(model=model, dataloader=dataloader, device=self.device)
        return {run.directory.name: scores}, labels


def compute_el2n_for_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    el2n_scores = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, file=sys.stdout, desc="Computing EL2N scores"):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            el2n_score = torch.norm(nn.functional.softmax(y_pred, dim=1) - nn.functional.one_hot(y), p=2, dim=1)
            el2n_scores.append(el2n_score)
            labels.append(y)
    return torch.cat(el2n_scores, dim=0).cpu().numpy(), torch.cat(labels, dim=0).cpu().numpy()


def compute_grand_for_model(model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            loss_fn: Callable = nn.CrossEntropyLoss()) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    assert dataloader.batch_size == 1, "Batch size must be 1 for computing the GraNd score."
    grand_scores = []
    labels = []
    for x, y in tqdm(dataloader, file=sys.stdout, desc="Computing GraNd scores"):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        model.zero_grad()
        loss.backward()
        grand_score = compute_grad_norm(model, ord=2)
        grand_scores.append(grand_score)
        labels.append(y)
    return np.array(grand_scores), torch.cat(labels, dim=0).cpu().numpy()


def _get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_result", type=str, required=True)
    parser.add_argument("--compute_at_progress_idx", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--compute_el2n", action="store_false")
    parser.add_argument("--compute_grand", action="store_false")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = _get_args()
    print('Computing EL2N and GraNd scores...')
    score_computer = El2nAndGrandScores(sweep_result=args['sweep_result'],
                                        compute_at_progress_idx=args['compute_at_progress_idx'],
                                        batch_size=args['batch_size'],
                                        gpu_id=args['gpu_id'],
                                        compute_el2n=args['compute_el2n'],
                                        compute_grand=args['compute_grand'])
    score_computer.compute()
    print(args)
    print('Done.')