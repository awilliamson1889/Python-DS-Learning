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

## How use DSReader
```bash
# First step, import module.

from src.reader import DSReader

# Step two, create an instance of the class

my_dataset = DSReader(data) # data - this path to the dataset in csv format

# IMPORTANT. The data set should be in the format:
# "email", label
# "message text"..., 1
# "email text"..., 0
# "Hello world", 0
# "something text", 1
# "something text two", 0
# ...
# "email text-n", label-n

# Step three, use class methods
# 5 steps to clean dataset
# to_lower() -> remove_digits() -> remove_punctuation_marks() -> remove_stopwords() -> remove_duplicates()

my_dataset.to_lower()
my_dataset.remove_digits()
my_dataset.remove_punctuation_marks()
my_dataset.remove_stopwords()
my_dataset.remove_duplicates()

```