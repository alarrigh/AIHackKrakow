#!/bin/bash

# install explainability widgets
conda activate py35
pip install --upgrade azureml-sdk[contrib]==1.0.65
jupyter nbextension install --py --sys-prefix azureml.contrib.explain.model.visualize
jupyter nbextension enable --py --sys-prefix azureml.contrib.explain.model.visualize