# %%
import torch
import hydra
from torch.utils import data
from omegaconf import OmegaConf, DictConfig
from ml_utilities.output_loader.result_loader import JobResult
from ml_utilities.data.datasetgenerator import DatasetGenerator
from tflearning.sample_difficulty import prediction_depth as pdepth
import timm
# %%
@hydra.main(version_base=None, config_path='configs_example_difficulty', config_name='config')
def main(cfg: DictConfig) -> None:
  device = torch.device('cuda:1')

  # %%
  # create dataset cf10 train split
  data_cfg = """
  data:
    dataset: cifar10
    dataset_kwargs:
      data_root_path: /system/user/beck/pwbeck/data
    dataset_split:
      train_val_split: 0.9
      restrict_n_samples_train_task: 10000
  """
  data_cfg = OmegaConf.create(data_cfg)
  cf10_ds_gen = DatasetGenerator(**data_cfg.data)
  cf10_ds_gen.generate_dataset()
  cf10_train = cf10_ds_gen.train_split
  cf10_val = cf10_ds_gen.val_split

  cf10_trainloader = data.DataLoader(cf10_train, batch_size=128, shuffle=False)
  cf10_valloader = data.DataLoader(cf10_val, batch_size=128, shuffle=False)

  # %%
  resnet = timm.create_model('resnet18', pretrained=True)
  layer_names = pdepth.find_layer_names(resnet, 'act2')
  layer_names.append('act1') # add the last layer (activation) of the first block

  print('layer_names:', layer_names)

  # %%
  pred_depth = pdepth.PredictionDepth(resnet,
                                      layer_names,
                                      experiment_specifier=f'{data_cfg.data.dataset}_timmresnet-ground_truth_label',
                                      train_dataloader=cf10_trainloader,
                                      test_dataloader=cf10_valloader, 
                                      prediction_depth_mode='ground_truth_label', 
                                      device=device)
  
  plots = pred_depth.make_plots()
  # del pred_depth
  # pred_depth = pdepth.PredictionDepth(resnet,
  #                                     layer_names,
  #                                     experiment_specifier='cf10_myresnet-model_prediction',
  #                                     train_dataloader=cf10_trainloader,
  #                                     test_dataloader=cf10_valloader, 
  #                                     prediction_depth_mode='model_prediction')
  # plots = pred_depth.make_plots()


  print('Done.')

if __name__ == '__main__':
  main()
