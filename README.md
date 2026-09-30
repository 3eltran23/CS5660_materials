# CS 5660 Materials

Course materials and examples for CS 5660 Advanced Artificial Intelligence.

The `HyperparameterTunner` folder contains N-Queens evolutionary-algorithm examples using DEAP, including grid/random search and Optuna-based hyperparameter tuning.

## Create the Conda environment

Open a terminal and create a Python 3.12 environment named `CS5660`:

```bash
conda create --name CS5660 python=3.12 -y
conda activate CS5660
```

## Install the required packages

With the `CS5660` environment active, run:

```bash
python -m pip install numpy pandas matplotlib deap optuna plotly nbformat jupyterlab ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name CS5660 --display-name "Python (CS5660)"
```

## Run the notebooks

From the repository root, start JupyterLab:

```bash
conda activate CS5660
jupyter lab
```

Open one of the following notebooks and select the **Python (CS5660)** kernel:

- `HyperparameterTunner/Hyperparameter_tuning.ipynb`
- `HyperparameterTunner/Hyperparameter_tuning_Optuna.ipynb`

The notebooks import `ea.py` and `utils.py` from the same directory, so keep those files together with the notebooks.
