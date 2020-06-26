# ml-udacity-capstone
Working to classify the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) by breed using PyTorch and AWS Sagemaker. This project was completed as a capstone project for the [Udacity Machine Learning Engineer Nanodegree Program](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t). If you want good exposure to Convolutional Neural Networks and Transfer Learning, come on in!

Most of the core workflow is contained in the Jupyter notebooks, supported by subdirectories that contain `.py` modules for data pre-processing, training, and prediction.

## local software requirements
Create a python 3.6.10 virtual environment to mess around with if you're not working in a sagemaker notebook.Then, install your requirements. The example below creates a local python install and virtual environment using [pyenv and pyenvvirtualenvwrapper](https://gist.github.com/eliangcs/43a51f5c95dd9b848ddc), which is simply my preference.

```
pyenv virtualenv 3.6.10 doggo-viz
pyenv activate doggo-viz
pip install -r requirements.txt
```
These requirements should be more or less compatible with anything that comes in Sagemaker's miniconda distribution.
While this local requirements list **does** include the `sagemaker` SDK bindings, I didn't end up running any of the deployment and training code that you see in the Jupyter notebooks outside of a Sagemaker notebook instance. I experimented briefly with [sagemaker's local mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode) but couldn't decode all of the AWS terminology necessary for it to work. 

The only other local requirement that you will need is the [wget download utility](https://www.gnu.org/software/wget/), which may or may not be present. You can check to see if you have it with the first command below. If that fails to return a valid path, the easiest way to install it on OSX is with [brew](https://brew.sh/).

```
which wget
brew install wget
```

## sagemaker software requirements
Once running this repository in AWS sagemaker, the only additional python library I needed to install was `xmltodict`. I did this by opening a Terminal in Jupyter and running the following command

```
conda install -c conda-forge xmltodict
```
For reasons I don't fully understand, to be honest, I also had to upgrade the `sagemaker` and `boto3` libraries from within the notebook instance, because for some reason I wasn't running the latest verison. As a gut check against this, from within **whichever notebook I opened first in sagemaker** I recommend executing the following Python magic function

```
!pip install -U sagemaker boto3
```

Once you've done these two checks, you should be ready to go.

## acquiring the dataset
I generally feel it's bad practice to check data files into version control, so you won't find the raw images here. However, so long as you have wget present in your environment, you should be able to execute the initial cells of _any_ of the three Jupyter notebooks to bring local copies of the Stanford Dogs datset into a .gitignored `data` directory. Check there for more detail and guidance.
