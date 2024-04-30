# DIP Final Project - Scene segmentation, semantic segmentation

Authors: Ramon Asuncion Batista, Santiago Hernandez, Warren Wang

## Description

Scene segmentation has many use cases in many fields, including but not limited to autonomous driving purposes.

## Video Demonstration

[Video](https://www.youtube.com/watch?v=dSuW3t7GcoA)

## Data

We used the CityScapes dataset to train the model. You can download the dataset from [this link](https://www.cityscapes-dataset.com/); note you will need to create an account and prove you are affiliated with an educational institution (have a .edu email).

## Getting Started

Tested using Python 3.10.13.
Using your preferred python environment manager, install requisite packages from the requirements.txt file:

```
pip install -r requirements.txt
```

Webserver setup

```
cd webserver
npm install
```

Then to get this running, we have provided a bash script to run both the node webserver and the python flask api. After activating your python environment, execute the script `run.sh`.

### References

The model architecture, training, and inference code were based off of the tutorial code found [here](https://github.com/talhaanwarch/youtube-tutorials/blob/main/cityscape-tutorial.ipynb).
