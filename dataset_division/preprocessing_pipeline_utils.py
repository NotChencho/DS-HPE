# Libraries needed by functions pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def core_columns(linear):
  # Function that assigns the core columns for each case (regression/tree model).
  # Returns the list of said cols.
  core_cols = ['submit_time']
  if linear:
      core_cols.extend([
            'time_limit_cat', 'time_limit_scaled', 
            'num_nodes_req', 'has_req_nodes', 'num_cores_req', 'cores_per_task',
            'num_gpus_req', 'mem_req', 'has_req_threads_per_core', 'is_shared_job',
            'partition_final', 'qos_final'
        ])
  else:
      core_cols.extend([
            'time_limit', 'time_limit_cat',
            'num_nodes_req', 'has_req_nodes', 'num_cores_req', 'cores_per_task',
            'num_gpus_req', 'mem_req', 'has_req_threads_per_core', 'is_shared_job'
        ])
  # Temporal cols + target cols (node_power_consumption)    
  core_cols.extend([
        'dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6',
        'dom_1','dom_2','dom_3','dom_4','dom_5','dom_6','dom_7','dom_8','dom_9',
        'dom_10','dom_11','dom_12','dom_13','dom_14','dom_15','dom_16','dom_17',
        'dom_18','dom_19','dom_20','dom_21','dom_22','dom_23','dom_24','dom_25',
        'dom_26','dom_27','dom_28','dom_29','dom_30','dom_31',
        'hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6',
        'hour_7','hour_8','hour_9','hour_10','hour_11','hour_12',
        'hour_13','hour_14','hour_15','hour_16','hour_17','hour_18',
        'hour_19','hour_20','hour_21','hour_22','hour_23',
        'node_power_min', 'node_power_mean', 'node_power_max'
    ])
  
  return core_cols

def ordered_cols(df, linear, include_mem, include_cpu, group_option, num_tasks_option):
  # Funtion that returns the columns of the df ordered.
  # Takes into account the different options depending on the arguments.
  core_cols=core_columns(linear)
  
  if include_mem:
    cols_mem_target = ['mem_power_min', 'mem_power_mean', 'mem_power_max']
  else:
    cols_mem_target = []

  if include_cpu:
    cols_cpu_target = ['cpu_power_min', 'cpu_power_mean', 'cpu_power_max']
  else:
    cols_cpu_target = []
  
  if group_option == 2:
    cols_group=['group']
  elif group_option == 5:
    cols_group=['group_id_enc']
  else:
    cols_group=['user_id_enc']

  if num_tasks_option in (1, 2):
    cols_num_task=['num_tasks_final', 'num_tasks_missing_or_inconsistent']
  else:
      cols_num_task = []
  
  ordered_cols = (
    cols_group
    + cols_num_task
    + core_cols
    + cols_mem_target
    + cols_cpu_target
)
  return df[ordered_cols]

# Functions used to calculate min, mean, max of a target column, taking into
# account the possibility that the power_consumtion list is empty, to avoid errors.
def safe_min(x):
    return np.min(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan

def safe_mean(x):
    return np.mean(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan

def safe_max(x):
    return np.max(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan