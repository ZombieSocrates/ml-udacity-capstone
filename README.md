# ml-udacity-capstone
Working with the Stanford Dogs dataset with PyTorch

## local setup
Create a python 3.6.10 virtual environment to mess around with if you're not working in a sagemaker notebook.Then, install your requirements. These should be more or less compatible with anything that comes in Sagemaker's miniconda distribution.

```
pyenv virtualenv 3.6.10 doggo-viz
pyenv activate doggo-viz
pip install -r requirements.txt
```

Also, just FYI, these requirements won't include any PyTorch and Sagemaker libraries, because I'm assuming most of those things will happen in the AWS notebooks.
