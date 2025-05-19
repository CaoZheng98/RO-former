import pandas as pd
from pathlib import Path
from configs.project_config import ProjectConfig
import os

class JSDatasetConfig:
    #JieShou
    outfall_names = ['DCHTLN', 'DCHWSC', 'HMGDX', 'HMGSM', 'HMGWS', 'JBHDT', 'JBHWZ', 'JHHHHC', 'JHHXY', 'JLH',
                     'WFG', 'XFG']
    train_rate = 0.7
    test_rate = 0.2
    rates = [train_rate, test_rate]
    project_root = ProjectConfig.project_root
    target = ['实时流量(m³/h)']   # ['溶解氧(mg/L)']#     # '氨氮(mg/L)',
    # outfall_names = ['DCHTLN']  #['DCHTLN', 'DCHWSC', 'HMGDX', 'HMGSM', 'HMGWS', 'JBHDT', 'JBHWZ', 'JHHHHC', 'JHHXY', 'JLH', 'WFG', 'XFG']
    dataset_info = f"{outfall_names}_{train_rate}_{test_rate}_{target}"
