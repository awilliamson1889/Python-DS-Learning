[![Pandas-Test](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pandas_test.yml/badge.svg)](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pandas_test.yml)
[![Numpy-Test](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/numpy_test.yml/badge.svg)](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/numpy_test.yml)
[![Pylint-Check](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pylint_check.yml/badge.svg)](https://github.com/awilliamson1889/Python-DS-Learning/actions/workflows/pylint_check.yml)

# Python-DS-Learning

#Models scores
<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>model</th>      <th>accuracy</th>      <th>precision</th>      <th>recall</th>      <th>F1-score</th>    </tr>  </thead>  <tbody>    <tr>      <th>3</th>      <td>MultinomialNB1</td>      <td>0.963</td>      <td>0.961</td>      <td>0.897</td>      <td>0.928</td>    </tr>    <tr>      <th>5</th>      <td>MultinomialNB3 - tokenizer</td>      <td>0.962</td>      <td>0.959</td>      <td>0.896</td>      <td>0.926</td>    </tr>    <tr>      <th>7</th>      <td>SGDClassifier1 + chi</td>      <td>0.954</td>      <td>0.903</td>      <td>0.937</td>      <td>0.92</td>    </tr>    <tr>      <th>8</th>      <td>SGDClassifier2 + tfidf</td>      <td>0.956</td>      <td>0.957</td>      <td>0.879</td>      <td>0.916</td>    </tr>    <tr>      <th>0</th>      <td>RandomForestClassifier1 + tfidf</td>      <td>0.952</td>      <td>0.94</td>      <td>0.88</td>      <td>0.909</td>    </tr>    <tr>      <th>2</th>      <td>RandomForestClassifier2 + chi</td>      <td>0.944</td>      <td>0.907</td>      <td>0.894</td>      <td>0.9</td>    </tr>    <tr>      <th>1</th>      <td>RandomForestClassifier1 + tfidf - tokenizer</td>      <td>0.938</td>      <td>0.893</td>      <td>0.896</td>      <td>0.894</td>    </tr>    <tr>      <th>6</th>      <td>MultinomialNB4 + tfidf - tokenizer</td>      <td>0.888</td>      <td>0.997</td>      <td>0.569</td>      <td>0.725</td>    </tr>    <tr>      <th>4</th>      <td>MultinomialNB2 + tfidf</td>      <td>0.881</td>      <td>0.997</td>      <td>0.543</td>      <td>0.703</td>    </tr>  </tbody></table>
Since we need to minimize NON-SPAM emails getting into the SPAM folder, Precision is an important parameter when evaluating models. After comparing all the models, the most successful in our case was MultinomialNB1.

*test size = 20%

## Project setup

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
pylint src/reader.py
```
## Run fit & predict process 
```bash
python email-classifier.py
```
