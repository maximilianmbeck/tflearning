# %%
from torch.utils import data

from omegaconf import OmegaConf
from ml_utilities.output_loader.result_loader import JobResult
from ml_utilities.data.datasetgenerator import DatasetGenerator
from tflearning.sample_difficulty import prediction_depth as pdepth

# %%
job_cf10resnet = JobResult('/system/user/beck/pwbeck/projects/regularization/tflearning/outputs/IA-A-cifar10-17.2.2-resnet-B--230204_215821')
job_cf10resnet 

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
myresnet = job_cf10resnet.get_model_idx(-1)
layer_names = pdepth.find_layer_names(myresnet, 'act2')
layer_names.append('resnet.0.2') # add the last layer (activation) of the first block

print('layer_names:', layer_names)

# %%
pred_depth = pdepth.PredictionDepth(myresnet,
                                    layer_names,
                                    experiment_specifier='cf10_myresnet-ground_truth_label',
                                    train_dataloader=cf10_trainloader,
                                    test_dataloader=cf10_valloader, 
                                    prediction_depth_mode='ground_truth_label')
plots = pred_depth.make_plots()
del pred_depth
pred_depth = pdepth.PredictionDepth(myresnet,
                                    layer_names,
                                    experiment_specifier='cf10_myresnet-model_prediction',
                                    train_dataloader=cf10_trainloader,
                                    test_dataloader=cf10_valloader, 
                                    prediction_depth_mode='model_prediction')
plots = pred_depth.make_plots()


print('Done.')