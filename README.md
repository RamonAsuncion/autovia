# Digital Image Processing Final Project - Semantic Segmentation

Authors: Ramon Asuncion Batista, Santiago Hernandez, Warren Wang

## Description

Scene segmentation has many use cases in many fields, including but not limited to autonomous driving purposes.
In this repository we provide a simple web interface for automatic segmentation of roads into predefined classes used in the CityScapes dataset. As can be seen in the video demonstration below, the model performs moderately well on out-of-distribution data (any roads outside of the roads that was used in the CityScapes data). 

## Video Demonstration

[![video](https://img.youtube.com/vi/dSuW3t7GcoA/1.jpg)](https://www.youtube.com/watch?v=dSuW3t7GcoA)

## Data

We used the CityScapes dataset to train the model. You can download the dataset from [this link](https://www.cityscapes-dataset.com/); note you will need to create an account and prove you are affiliated with an educational institution (have a .edu email).

## Model Weights

We provide the model weights ready for download for the fine-tuned resnet34 imagenet pytorch segmentation pretrained model on the CityScapes dataset [here](https://drive.google.com/file/d/1W7VkRgNnAAoXi5Y6FjBWOggHTkfTu5aT/view?usp=sharing). In our use, we put the model weights in at `./weights/` as can be seen in the `run.sh` bash script provided; feel free to place your model weights file in the same place (you will need to create that directory).

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
