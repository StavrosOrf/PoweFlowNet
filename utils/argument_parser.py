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
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--case', type=str, default='14', help='Grid case')
    parser.add_argument('--wandb', default=True, help='Enable wandb logging',action=argparse.BooleanOptionalAction)
    parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction)
    
    
    args, left_argv = parser.parse_known_args()
    
    return args