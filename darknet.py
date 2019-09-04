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