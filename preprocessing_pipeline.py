
# Libraries needed by functions pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from preprocessing_pipeline_utils import ordered_cols, safe_min, safe_mean, safe_max

# Pipeline function
def pipeline_features(df, linear= True, include_cpu=False, include_mem=False, group_option=2, num_tasks_option=1):
  """Function that loads the data of Marconi100 and makes the appropriate transformations
     in order to return the df needed for the regression task.
     Arguments:
     - linear= True if the model to be applied is linear/NN regression. False if it is a tree model.
     - include_cpu: True if we want to include cpu power consumption in the model. False otherwise.
     - include_mem: True if we want to include mem power consumption in the model. False otherwise.
     - group_option: int, possible values {2: use main and others, 5: use all groups, 0: do not include group_id}
     - num_task_option: int, possible values 
     {0: do not include num_tasks, 1: use calculated num_tasks for NAs, 2: use calculated num_tasks for NAs and wrongly calculated entries}

     Returns: df with the features needed for the model.
     """
  # Read the Marconi 100 data (job table)
  df = pd.read_parquet("job_table.parquet")

  # Standard transformation over times to ensure they are formated properly.
  df["submit_time"]   = pd.to_datetime(df["submit_time"],   utc=True, errors="coerce")

  # All num_task are integers → convert column to integer type
  # Ensure it's numeric
  df["num_tasks"] = df["num_tasks"].astype("Int64")

  # TRANSFORMATIONS OVER FEATURES
  # partition
  df["partition_final"] = df["partition"].replace({0: "__other__", 2: "__other__"}).astype("category")

  # qos
  main_qos = ["1", "4", "8", "11"]
  df["qos_final"] = df["qos"].astype(str).where(df["qos"].isin(main_qos), "__other__").astype("category")

  df = df.drop(columns=["partition", "qos"])

  # time_limit
  # Add to df bins that classify the jobs regarding their time_limit
  # Added because the may be useful at interpretation time (when analyzing feature importance)
  bins = [0, 60, 300, 720, 1440]
  labels = ["short (<1h)", "medium (1–5h)", "long (5–12h)", "very long (~1d)"]
  df["time_limit_cat"] = pd.cut(df["time_limit"], bins=bins, labels=labels, include_lowest=True)

  # Also, take the Log-transform to reduce skew (can be useful for linear regressioin models)
  df["log_time_limit"] = np.log1p(df["time_limit"])

  # take also the scaled value (can be useful for linear/NN models)
  scaler = StandardScaler()
  df["time_limit_scaled"] = scaler.fit_transform(df[["log_time_limit"]])

  # req_nodes
  df["has_req_nodes"] = df["req_nodes"].notna().astype(int)
  df = df.drop(columns=["req_nodes", "nodes"])

  # threads_per_core
  df["has_req_threads_per_core"] = df["threads_per_core"].notna().astype(int)
  df = df.drop(columns=["threads_per_core"])

  # shared
  df["is_shared_job"] = df["shared"].map({
      "0": 0,
      "OK": 1,
      "USER": 1
  }).astype(int)

  df = df.drop(columns=["shared"])

  # OPTIONS FOR GROUP_ID
  if group_option == 2:
    print("Option leave two groups: main, other.")
    # Encode group keeping main group, others
    main_group = 25200
    df["group"] = np.where(df["group_id"] == main_group, "main", "other")
  elif group_option == 5:
    print("Option leave all groups, encoded.")
    # If we want to keep group_id without contracting it
    le_group = LabelEncoder()
    df["group_id_enc"] = le_group.fit_transform(df["group_id"])
    # df = df.drop(columns=["user_id"])
  else: 
    # To keep user_id in models when we want to capture fine grain patterns or to
    # study user specific habits
    print("Option eliminate group_id, keep user_id encoded.")
    # If we want to eliminate group_id
    le_user = LabelEncoder()
    df["user_id_enc"] = le_user.fit_transform(df["user_id"])
  df = df.drop(columns=["user_id", "group_id"])
  
  # OPTIONS FOR NUM_TASKS
  # Calculate derived number of tasks
  df["num_tasks_calc"] = (df["num_cores_req"] // df["cores_per_task"]).astype("Int64")

  # Create flag for missing or inconsistent num_tasks
  df["num_tasks_missing_or_inconsistent"] = (
      df["num_tasks"].isna() |
      (df["num_tasks"] * df["cores_per_task"] != df["num_cores_req"])
  )

  if num_tasks_option == 1:
    print("Option Impute NAs, leave incorrect num_tasks entries.")
    # Fill missing values (NAs) with calculated value, do not correct wrongly calculated num_tasks
    # Create calc_final column where to store the value to use of num_tasks.Init with raw value.
    df["num_tasks_final"] = df["num_tasks"]
    # For NA values, substitute by calculated value. For incorrect values, keep it.
    df.loc[df["num_tasks"].isna(), "num_tasks_final"] = df.loc[df["num_tasks"].isna(), "num_tasks_calc"]
    df = df.drop(columns=["num_tasks", "num_tasks_calc"]) # Leave num_tasks_final
  elif num_tasks_option == 2:
    print("Option Impute NAs, correct incorrect num_tasks entries.")
    # Fill missing values (NAs) with calculated value
    df["num_tasks_final"] = df["num_tasks_calc"]
    df = df.drop(columns=["num_tasks", "num_tasks_calc"]) # Leave num_tasks_final
  else:
    print("Option eliminate num_tasks.")
    df = df.drop(columns=["num_tasks", "num_tasks_calc", "num_tasks_missing_or_inconsistent"]) # Do not keep num_tasks in df

  # OHE for submit_time
  # One-hot encode the day of the week: dow -> day of the week
  # Extract day of week (0=Mon, 6=Sun)
  df["submit_dayofweek"] = df["submit_time"].dt.dayofweek

  # One-hot encode
  df = pd.get_dummies(df, columns=["submit_dayofweek"], prefix="dow")

  # One-hot encode the day of the month: dom -> day of the month
  df["submit_day"] = df["submit_time"].dt.day
  df = pd.get_dummies(df, columns=["submit_day"], prefix="dom")

  # One-hot encode the hour
  df["submit_hour"] = df["submit_time"].dt.hour
  df = pd.get_dummies(df, columns=["submit_hour"], prefix="hour")

  # FEATURES TO BE DROPPED
  cols_to_drop_times=['eligible_time', 'start_time', 'end_time', 'run_time']
  cols_to_drop_alloc = ['num_nodes_alloc', 'num_cores_alloc', 'shared', 'num_gpus_alloc',
                'mem_alloc','cores_alloc_layout', 'cores_allocated']
  cols_to_drop_job= ['job_id','job_state', 'state_reason']
  cols_to_drop_others = ['req_switch', 'priority', 'derived_ec']

  cols_to_drop = (
      cols_to_drop_times
      + cols_to_drop_alloc
      + cols_to_drop_job
      + cols_to_drop_others
  )

  df = df.drop(columns=cols_to_drop, errors="ignore")

  if linear:
    print("Option generate DS for regression models.")
    pd.get_dummies(df, columns=["partition_final", "qos_final"], drop_first=True)
    df = df.drop(columns=["time_limit", "log_time_limit"])
  else:
    print("Option generate DS for tree models.")
    LabelEncoder().fit_transform(df["partition_final"])
    LabelEncoder().fit_transform(df["qos_final"])
    df = df.drop(columns=["partition_final", "qos_final"])
    df = df.drop(columns=["log_time_limit", "time_limit_scaled"])
    
  # Lastly, substitute the power consumption lists in the df by the target features: min, mean, max
  # Create new columns
  df["node_power_min"]  = df["node_power_consumption"].apply(safe_min)
  df["node_power_mean"] = df["node_power_consumption"].apply(safe_mean)
  df["node_power_max"]  = df["node_power_consumption"].apply(safe_max)
  # Drop the original list column
  df = df.drop(columns=["node_power_consumption"])
  
  if include_mem:
    # Create new columns
    print("Option include mem_power_consumtion as target.")
    df["mem_power_min"]  = df["mem_power_consumption"].apply(safe_min)
    df["mem_power_mean"] = df["mem_power_consumption"].apply(safe_mean)
    df["mem_power_max"]  = df["mem_power_consumption"].apply(safe_max)
    # Drop the original list column
    df = df.drop(columns=["mem_power_consumption"])

  if include_cpu:
    # Create new columns
    print("Option include cpu_power_consumtion as target.")
    df["cpu_power_min"]  = df["cpu_power_consumption"].apply(safe_min)
    df["cpu_power_mean"] = df["cpu_power_consumption"].apply(safe_mean)
    df["cpu_power_max"]  = df["cpu_power_consumption"].apply(safe_max)
    # Drop the original list column
    df = df.drop(columns=["cpu_power_consumption"])
  # Order the columns
  df = ordered_cols(df, linear, include_mem, include_cpu, group_option, num_tasks_option)
  return df


if __name__ == "__main__":
    # Read the Marconi 100 data (job table)
    df = pd.read_parquet("job_table.parquet")
    df_final= pipeline_features(df, linear= True, include_cpu=False, include_mem=False, group_option=2, num_tasks_option=1)
    
        # TEST AREA 
    print("Number of rows:", len(df_final))
    print("Number of columns:", len(df_final.columns))
    print("\nColumn names:", df_final.columns.tolist())
    print("\nFirst 5 rows:")
    print(df_final.head())