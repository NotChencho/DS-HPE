# DS-HPE
Proyecto en Ciencia de Datos


## Project Structure

```
├── data/
│
├── demo/
│   ├──
│   └──
│...
├── preprocessing/      # Data Engineering Pipeline
│   ├── dataset_creation_automl.ipynb
│   ├── prepare_ds_xgboost.ipynb
│   ├── Preprocessing_Marconi_DS.ipynb
│   ├── preprocessing_pipeline_utils.py
│   └── preprocessing_pipeline.py
│
├── automl/                                             
│   ├── automl_dnn.ipynb
│   └── automl_xgboost.ipynb
│
├── main/                                               # Core model development
│   ├── NN/
│   │   ├── models/
│   │   │   ├── code_models/ # Model definitions & utilities
│   │   │   │   ├── sklearn_models.py
│   │   │   │   ├── torch_models.py                            
│   │   │   │   ├── train_sweep.py 
│   │   │   │   ├── sweep.yaml  # Hyperparameter tuning
│   │   │   │   └── training_utils.py
│   │   │   │
│   │   │   └── trained_models/     # Saved PyTorch (.pt) weights
│   │   │       └── app_model_attention.pt, app_model.pt, model.pt
│   │   │   
│   │   ├── model_interpretation.ipynb
│   │   └── training.ipynb
│   │
│   └── XGBOOST/                
│       ├── models/
│       │   ├── code_models/                            # Sweep configs and training scripts
│       │   │   ├── train_xgb_sweep.py
│       │   │   └── sweep.yaml
│       │   │
│       │   └── trained_models/                         # Saved XGBoost models
│       │
│       └── evaluate_model_xgboost.ipynb
│
├── .gitignore
│
└── README.md
```
