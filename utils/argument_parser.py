import argparse
import os
import json

def argument_parser():
    # config_parser = argparse.ArgumentParser(description='Argument parser for the project')
    parser = argparse.ArgumentParser(
        prog='PowerFlowNet',
        description='train neural network for power flow approximation'
    )
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    
    args, left_argv = parser.parse_known_args()
    
    return args