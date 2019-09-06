import torch
import torch.nn as nn
import numpy as np


def parse_config_file(cfgfile):
    """ This function is used to parse the yolov3 config file

        The config file contains information regarding the architecture of the model to be developed.
        Thus, the plan is to divide and segregate this each layer information provided in the config file
        into meaningful blocks of data. In this case, each layer information will be stored in a corresponding dict

        Returns a list of blocks of data present in config file
    """

    file = open(cfgfile, 'r')

    lines=file.read().split('\n')  # Reading each line and splitting it according to a newline

    print(len(lines))

    """ We have to remove the following items from the list:
        1. Empty Lines
        2. Commented lines
        3. Whitespaces?
    """

    lines = [line for line in lines if line]  # Removing empty lines
    lines = [line for line in lines if line[0]!="#"]  # Removing all lines which have the first character as a comment
    lines = [line.strip() for line in lines]  # Removing all leading and trailing whitespaces

    """ Converting the lines of config processed into blocks.
        Each new block is indicated by the type of the block which is specified by [<type>] format
        Each block contains values of required components.
    """

    block = {}  # Dictionary
    blocks = []  # List of dictionaries

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:  # Checking if the previously stored block is empty or not.
                blocks.append(block)
                block={}
            block["type"]=line.strip()  # Change it to 1:-1 ?
        else:
            # split by =
            key, value = line.split("=")
            block[key.strip()]=value.strip()
    blocks.append(block)

    return blocks


def create_neural_network(blocks):
    """The config file - cfg/yolov3.cfg has the following types of blocks, net, convolution, shortcut, upsample, route, yolo
        When values are set, they must be done so keeping in mind, the types of blocks displayed above.
    """

    previous_filters = 3  # starts out as 3, since we have a RGB image with 3 colour channels

    for block in blocks:
        module=nn.Sequential()
        if block[type]=="net":
            continue
        # Adding convolutional layer
        elif block[type]=="convolutional":
            activation=block["activation"]
            kernel_size=block["size"]
            pad=block["pad"]
            stride=block["stride"]
            filters=block["filter"]

            # Generally, bias does not exist in layers that have batch normalization.
            # This is so because batch normalization inherently takes care of the necessary bias factor.
            # Thus, we check below if the block defined has batch normalization and make a decision on bias accordingly.

            if "batch_normalize" in block.keys():
                bias=False
                batch_normalization=block["batch_normalize"]
            else:
                bias=True
                batch_normalization=0

            # Adding layers now
            conv_layer=nn.Conv2d(previous_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(conv_layer)

            # Add Batch Normalization
            if batch_normalization:
                bn_layer=nn.BatchNorm2d(filters)
                module.add_module(bn_layer)

            # Add Activation layer
            # Activation layer can be either a Leaky-Relu layer or a linear activation layer.
            if activation=="leaky":
                activation_layer=nn.LeakyReLU(0.1, inplace=True)
