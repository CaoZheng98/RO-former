import torch.nn as nn
import importlib

from ..project_config import ProjectConfig
from ..JSdataset_config import JSDatasetConfig
from ..data_shape_config import DataShapeConfig
from utils import nseloss


class PretrainLearningConfig:
    loss_type = "MSE"  # TODO: loss function type, chose in ["NSELoss" ,"MSE"]
    loss_functions = {"MSE": nn.MSELoss(), "NSELoss": nseloss.NSELoss()}
    loss_func = loss_functions[loss_type]

    scale_factor = 1  # TODO: usually, the bath_size bigger is, the learning_rate larger will have to be.
    n_epochs = 300  # TODO
    batch_size = 32   # TODO
    learning_rate = 0.001   # TODO
    # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
    scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "none"}
    # scheduler_paras = {"scheduler_type": "exp_decay", "decay_epoch": n_epochs * 0.5, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "cos_anneal", "cos_anneal_t_max": 32}

    learning_config_info = f"{loss_type}_n{n_epochs}_bs{batch_size}_lr{learning_rate}_{scheduler_paras['scheduler_type']}"


class JSTrain_BSConfig(PretrainLearningConfig):
    seed = 111111  # Random seed
    used_model = "Transformer"  # TODO

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode
    model_info = used_ModelConfig.model_info

    pre_train_id = f"{JSDatasetConfig.outfall_names}@{DataShapeConfig.data_shape_info}" \
                   f"@{JSDatasetConfig.train_rate}"
    pre_val_id = f"{JSDatasetConfig.outfall_names}@{DataShapeConfig.data_shape_info}"
    pre_test_id = f"{JSDatasetConfig.outfall_names}@{DataShapeConfig.data_shape_info}" \
                  f"@{JSDatasetConfig.test_rate}"
    final_train_data_path = ProjectConfig.final_data_root / f"{pre_train_id}_serialized_train.pkl"
    final_val_data_path = ProjectConfig.final_data_root / f"{pre_val_id}_serialized_val.pkl"
    final_test_data_path = ProjectConfig.final_data_root / f"{pre_test_id}_serialized_test.pkl"
    JS_train_config = {
        "project_root": JSDatasetConfig.project_root,
        "rates": JSDatasetConfig.rates,
        "final_data_path": final_train_data_path,
        "target": JSDatasetConfig.target,
        "outfall_names": JSDatasetConfig.outfall_names

    }

    JS_val_config = {
        "project_root": JSDatasetConfig.project_root,
        "rates": JSDatasetConfig.rates,
        "final_data_path": final_train_data_path,
        "target": JSDatasetConfig.target,
        "outfall_names": JSDatasetConfig.outfall_names
    }

    JS_test_config = {
        "project_root": JSDatasetConfig.project_root,
        "rates": JSDatasetConfig.rates,
        "final_data_path": final_train_data_path,
        "target": JSDatasetConfig.target,
        "outfall_names": JSDatasetConfig.outfall_names
    }

    saving_message = f"{model_info}@{JSDatasetConfig.dataset_info}@{DataShapeConfig.data_shape_info}" \
                     f"@{PretrainLearningConfig.learning_config_info}@seed{seed}"


    saving_root = ProjectConfig.run_root / saving_message
