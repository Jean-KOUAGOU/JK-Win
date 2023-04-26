# JK-Win
Repository for the Bremen Data Challenge Competition 2023


### Download ready-to-run

This GitHub repository serves for the description of the work and the steps to reproduce the results. It does not contain the data.

### Installation

1- Open a terminal then perform the following steps

- Create a conda environment ```conda create -n jk-win python=3.8```
- Activate the environment ```conda activate jk-win```
- Install required libraries ```pip install matplotlib pandas torch scikit-learn tqdm numpy```

2- Reproduce the reported results.

- First install jupyter-lab if necessary. This will allow you to run ```inference.ipynb``` to reproduce the reported results.

- Make sure the packages installed in 1- are also installed in the jupyter kernel on which ```inference.ipynb``` runs. For example you can run the following to install the required kernel:
i) ```conda activate jk-win```
ii) ```conda install ipykernel```
iii) ```python -m ipykernel install --user --name jk-win --display-name "jk-win"```

- Open ```inference.ipynb``` and select the kernel ```jk-win```

- Now run the notebook


### Main ideas of the work

- I compute time embeddings (with 64 dimensions) from 1950 to 2050 using the Date2Vect github repository https://github.com/ojus1/Time2Vec-PyTorch. This can be achieved by running ```Experiment.py``` in Date2Vec.

- I compute Sine and Cosine representations for each date and time; this yields a vector of dimension 4 for each date-time input. They are saved as ```data_train_additional.csv``` and ```data_test_additional.csv```. Use the notebook ```simple-preprocess.ipynb``` for this purpose.

- I concatenate the two types of vectors and obtain 68 dimensions.

- I perform a two step missing value fill-in via polynomial interpolation: 1) first interpolate NO2, NO3, NH4 using Temperatur and Salinitat, 2) interpolate SECCI and NOx using NO2, NO3, NH4  (see ```inference.ipynb``` for details)

- I implement 4 neural network architectures MLP, LSTM, GRU, SetTransformer (see models.py) and train them on the data in a 5-fold cross-validation setting. This can be achieved by running ```train.py```.

### Submission files

They can be found in the folder ```submissions```. The best public submission is ```prediction11.csv```. You can also analyze the other submissions 1 to 15.