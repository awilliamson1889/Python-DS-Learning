[![Pandas-Test](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pandas_test.yml/badge.svg)](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pandas_test.yml)
[![Pylint-Check](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pylint_check.yml/badge.svg)](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pylint_check.yml)

# Python-DS-Learning
 
Assume you already have Python and Miniconda installed, and your virtual environment activated.
## Preparation Linux

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Preparation Windows

```bash
py -m pip install -U pip
py -m pip install -r requirements.txt
```

## Run tests
```bash
python -m unittest tests/test_np.py
python -m unittest tests/test_pd.py
```
## Run pylint
```bash
pylint src\reader.py
```
