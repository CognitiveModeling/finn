
# FInite volume Neural Network (FINN)

This repository contains the PyTorch code for models, training, and testing, and Python code for data generation to conduct the experiments as reported in the work [Composing Partial Differential Equations with Physics-Aware Neural Networks](https://arxiv.org/abs/2111.11798)

If you find this repository helpful, please cite our work:

```
@inproceedings{karlbauer2021composing,
	address = {Baltimore, USA},
	author = {Karlbauer, Matthias and Praditia, Timothy and Otte, Sebastian and Oladyshkin, Sergey and Nowak, Wolfgang and Butz, Martin V},
	booktitle = {Proceedings of the 39th International Conference on Machine Learning},
	month = {16--23 Jul},
	series = {Proceedings of Machine Learning Research},
	title = {Composing Partial Differential Equations with Physics-Aware Neural Networks},
	year = {2022}
}
```

## Dependencies

We recommend setting up an (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) environment with python 3.7 (i.e. `conda create -n finn python=3.7`). The required packages for data generation and model evaluation are

  - `conda install -c anaconda numpy scipy`
  - `conda install -c pytorch pytorch==1.9.0`
  - `conda install -c jmcmurray json`
  - `conda install -c conda-forge matplotlib torchdiffeq jsmin`
  
Alternatively, the `environment.yml` can be used to create an according conda environment with `conda env create -f environment.yml`.

## Models & Experiments

The code of the different pure machine learning models ([Temporal Convolutional Network (TCN)](https://arxiv.org/pdf/2111.07470.pdf), [Convolutional Long-Short Term Memory (ConvLSTM)](https://arxiv.org/pdf/1506.04214.pdf)), physics-motivated models ([DISTANA](https://arxiv.org/pdf/1912.11141.pdf), CNN-NODE, [Fourier Neural Operator (FNO)](https://arxiv.org/pdf/2010.08895.pdf)) and physics-aware models ([Physics-Informed Neural Networks (PINN)](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [PhyDNet](https://arxiv.org/pdf/2003.01460.pdf), [APHYNITY](https://arxiv.org/pdf/2010.04456.pdf), FINN) can be found in the `models` directory.

Each model directory contains a `config.json` file to specify model parameters, data, etc. Please modify the sections in the respective `config.json` files as detailed below (further information about data and model architectures is reported in the according data sections of the paper's appendices):

```
"training": {
	"t_stop": 150  // burger and allen-cahn 150, diff-sorp 400, diff-react 70
},

"validation": {
	"t_start": 150,  // burger, burger_2d, and allen-cahn 150, diff-sorp 400, diff-react 70
	"t_stop": 200  // burger, burger2d, and allen-cahn 200, diff-sorp 500, diff-react 100
},

"data": {
	"type": "burger",  // "burger", "burger_2d", diffusion_sorption", "diffusion_reaction", "allen_cahn"
	"name": "data_ext",  // "data_train", "data_ext", "data_test"
}

"model": {
  	"name": "burger"  // "burger", "burger_2d", diff-sorp", "diff-react", "allen-cahn"
	"field_size": [49],  // burger and allen-cahn [49], diff-sorp [26], burger_2d and diff-react [49, 49]
	... other settings to be specified according to the model architectures section in the paper's appendix
}
```


The actual models can be trained and tested by calling the according `python train.py` or `python test.py` scripts. Alternatively, `python experiment.py` can be used to either train or test n models (please consider the settings in the `experiment.py` script).

## Data generation

The Python scripts to generate the burger, burger_2d, diffusion-sorption, diffusion-reaction, and  allen-cahn data can be found in the `data` directory.

In each of the `burger`, `burger_2d`, `diffusion_sorption`, `diffusion_reaction`, and `allen-cahn` directories, a `data_generation.py` and `simulator.py` script can be found. The former is used to generate train, extrapolation (ext), or test data. For details about the according data generation settings of each dataset, please refer to the corresponding data sections in the paper's appendix.

## Uncertainty quantification

The scripts required for the uncertainty quantification part can be found in the `sampling` directory.

Here, we provide the experimental data in the `data_core1.xlsx`, `data_core2.xlsx`, and `data_core2_long.xlsx` files. Additionally, the retardation factor values of the fitted physical model are provided in `retardation_phys.txt`.

There are two options to perform uncertainty quantification in this work, namely [Variational Inference](https://arxiv.org/pdf/1505.05424.pdf) and Markov Chain Monte Carlo (MCMC).

Training and sampling using the Variational Inference method is provided in the `var_inf.py` script, and the collections of MCMC methods are provided in the `sampler.py` script, including the [Metropolis-Hastings](https://www.jstor.org/stable/2684568?seq=1), [Metropolis-Adjusted Langevin Algorithm (MALA)](https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/bj/1178291835.full), and [Barker proposal](https://arxiv.org/pdf/1908.11812.pdf).

Sampling with MCMC can be performed by running the `main.py` script, and selecting the desired sampler in the `config.json` file. In the config file, user can also choose whether to start with a random initial point or using a pre-trained model provided in the `checkpoints` directory (by setting `sampling.random_init` true or false).
 
# Citations
 
```

@article{espeholt2021skillful,
	title={Skillful Twelve Hour Precipitation Forecasts using Large Context Neural Networks},
	author={Espeholt, Lasse and Agrawal, Shreya and S{\o}nderby, Casper and Kumar, Manoj and Heek, Jonathan and Bromberg, Carla and Gazen, Cenk and Hickey, Jason and Bell, Aaron and Kalchbrenner, Nal},
	journal={arXiv preprint arXiv:2111.07470},
	year={2021}
}

@article{shi2015convolutional,
	title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
	author={Shi, X. and Chen, Z. and Wang, H. and Yeung, D.Y. and Wong, W.K. and Woo, W.C.},
	journal={arXiv preprint arXiv:1506.04214},
	year={2015}
}

@article{Karlbauer2019,
	title={A Distributed Neural Network Architecture for Robust Non-Linear Spatio-Temporal Prediction},
	author={Karlbauer, M. and Otte, S. and Lensch, H.P.A. and Scholten, T. and Wulfmeyer, V. and Butz, M.V.},
	year={2019},
	journal={arXiv preprint arXiv:1912.11141},
}

@article{li2020fourier,
	title={Fourier neural operator for parametric partial differential equations},
	author={Li, Z. and Kovachki, N. and Azizzadenesheli, K. and Liu, B. and Bhattacharya, K. and Stuart, A. and Anandkumar, A.},
	journal={arXiv preprint arXiv:2010.08895},
	year={2020}
}

@article{Raissi2019,
	author={Raissi, M. and Perdikaris, P. and Karniadakis, G.E.},
	journal={Journal of Computational Physics},
	title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
	year={2019},
	volume={378},
	pages={686-707},
	doi={10.1016/j.jcp.2018.10.045}
}

@inproceedings{guen2020disentangling,
	title={Disentangling physical dynamics from unknown factors for unsupervised video prediction},
	author={Guen, V.L. and Thome, N.},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	pages={11474--11484},
	year={2020}
}

@article{guen2020augmenting,
	title={Augmenting physical models with deep networks for complex dynamics forecasting},
	author={Yin, Y. and Guen, V.L. and Dona, J. and Ayed, I. and de B{\'e}zenac, E. and Thome, N. and Gallinari, P.},
	journal={arXiv preprint arXiv:2010.04456},
	year={2020}
}
 
@misc{https://doi.org/10.48550/arxiv.1505.05424,
	doi = {10.48550/ARXIV.1505.05424},
	url = {https://arxiv.org/abs/1505.05424},
	author = {Blundell, Charles and Cornebise, Julien and Kavukcuoglu, Koray and Wierstra, Daan},
	title = {Weight Uncertainty in Neural Networks},
	year = {2015}
}

@article{Chib1995MH,
	url = {http://www.jstor.org/stable/2684568},
	author = {Siddhartha Chib and Edward Greenberg},
	journal = {The American Statistician},
	number = {4},
	pages = {327--335},
	publisher = {[American Statistical Association, Taylor & Francis, Ltd.]},
	title = {Understanding the Metropolis-Hastings Algorithm},
	volume = {49},
	year = {1995}
}

@article{Roberts1996MALA,
	author = {Gareth O. Roberts and Richard L. Tweedie},
	title = {{Exponential convergence of Langevin distributions and their discrete approximations}},
	volume = {2},
	journal = {Bernoulli},
	number = {4},
	publisher = {Bernoulli Society for Mathematical Statistics and Probability},
	pages = {341 -- 363},
	year = {1996},
	doi = {bj/1178291835}
}

@misc{Livingstone2019Barker,
	doi = {10.48550/ARXIV.1908.11812},
	url = {https://arxiv.org/abs/1908.11812},
	author = {Livingstone, Samuel and Zanella, Giacomo},
	title = {The Barker proposal: combining robustness and efficiency in gradient-based MCMC},
	year = {2019}
}


```

# Code contributors
* [Matthias Karlbauer](https://github.com/MatKbauer)
* [Timothy Praditia](https://github.com/timothypraditia)