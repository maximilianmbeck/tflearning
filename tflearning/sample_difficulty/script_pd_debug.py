# %%
import sys

import hydra
import timm
import torch
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from tqdm import tqdm

from tflearning.data.creator import create_datasetgenerator
from tflearning.models.creator import create_model
from tflearning.sample_difficulty import prediction_depth as pdepth
from tflearning.sample_difficulty.prediction_depth_script import \
    PredictionDepthRunnerConfig


# %%
@hydra.main(version_base=None, config_path='configs_scripts_hydra', config_name='config')
def main(cfg: DictConfig) -> None:
  config = from_dict(data_class=PredictionDepthRunnerConfig, data=OmegaConf.to_container(cfg.config.kwargs, resolve=True))
  print(config)

  device = torch.device(f'cuda:{config.experiment_data.gpu_id}' if torch.cuda.is_available() else 'cpu')

  # data
  ds_gen = create_datasetgenerator(config.data)
  ds_gen.generate_dataset()
  
  # model
  model = create_model(model_cfg=config.model)


  dataset = ds_gen.train_split
  dataloader = data.DataLoader(dataset, batch_size=config.prediction_depth.batch_size, shuffle=False, num_workers=0)
  
  # create feature extractor
  feature_extractor = pdepth.LayerFeatureExtractor(model, [], config.prediction_depth.features_before, config.prediction_depth.append_softmax_output)
  feature_extractor.to(device)

  for layer_name in config.prediction_depth.layer_names:
    feature_extractor.layer_names = [layer_name]
    for x, y in tqdm(dataloader, file=sys.stdout, desc=f"Layer: {layer_name}"):
      with torch.no_grad():
          x, y = x.to(device), y.to(device)
          y_pred, feats = feature_extractor(x)
          print(feats.keys())

  print('Done.')

if __name__ == '__main__':
  main()
