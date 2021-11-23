
# FInite volume Neural Network (FINN)

This repository contains the PyTorch code for models, training, and testing, and Python code for data generation to conduct the experiments as reported in the work [Composing Partial Differential Equations with Physics-Aware Neural Networks](...)

If you find this repository helpful, please cite our work:

```
@article{...,
	author    = {...},
	title     = {Composing Partial Differential Equations with Physics-Aware Neural Networks},
	journal   = {...},
	year      = {...},
}
```

## Dependencies

We recommend setting up an (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) environment with python 3.7 (i.e. `conda create -n finn python=3.7`). The required packages for data generation and model evaluation are

  - `conda install -c anaconda numpy scipy`
  - `conda install -c pytorch pytorch==1.9.0`
  - `conda install -c jmcmurray json`
  - `conda install -c conda-forge matplotlib torchdiffeq jsmin`

## Models & Experiments

The code of the different pure machine learning models (TCN, ConvLSTM, DISTANA) and physics-aware models (PINN, PhyDNet, FINN) can be found in the `models` directory.

Each model directory contains a `config.json` file to specify model parameters, data, etc. Please modify the sections in the respective `config.json` files as detailed below (further information about data and model architectures is reported in the according data sections of the paper's appendices):

```
"training": {
	"t_stop": 150  // burger and allen-cahn 150, diff-sorp 400, diff-react 70
},

"validation": {
	"t_start": 150,  // burger and allen-cahn 150, diff-sorp 400, diff-react 70
	"t_stop": 200  // burger and allen-cahn 200, diff-sorp 500, diff-react 100
},

"data": {
	"type": "burger",  // "burger", "diffusion_sorption", "diffusion_reaction", "allen_cahn"
	"name": "data_ext",  // "data_train", "data_ext", "data_test"
}

"model": {
  	"name": "burger"  // "burger", "diff-sorp", "diff-react", "allen-cahn"
	"field_size": [49],  // burger and allen-cahn [49], diff-sorp [26], fhn [49, 49]
	... other settings to be specified according to the model architectures section in the paper's appendix
}
```


The actual models can be trained and tested by calling the according `python train.py` or `python test.py` scripts. Alternatively, `python experiment.py` can be used to either train or test n models (please consider the settings in the `experiment.py` script).

## Data generation

The Python scripts to generate the burger, diffusion-sorption, diffusion-reaction, and  allen-cahn data can be found in the `data` directory.

In each of the `burger`, `diffusion_sorption`, `diffusion_reaction`, and `allen-cahn` directories, a `data_generation.py` and `simulator.py` script can be found. The former is used to generate train, extrapolation (ext), or test data. For details about the according data generation settings of each dataset, please refer to the corresponding data sections in the paper's appendices.
