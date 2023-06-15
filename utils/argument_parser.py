import argparse
import os
import json

def argument_parser():
    # config_parser = argparse.ArgumentParser(description='Argument parser for the project')
    config_parser = argparse.ArgumentParser(
        prog='PowerFlowNet',
        description='parse json configs',
        add_help=False) # a must because otherwise the child will have two help options
    config_parser.add_argument('--cfg_json','--config','--configs', default='configs/standard.json',type=str)

    parser = argparse.ArgumentParser(parents=[config_parser])
    
    parser = argparse.ArgumentParser(
        prog='PowerFlowNet',
        description='train neural network for power flow approximation'
    )
    
    # Network Parameters
    parser.add_argument('--nfeature_dim', type=int, default=6, help='Number of node features')
    parser.add_argument('--efeature_dim', type=int, default=2, help='Number of edge features')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden features')
    parser.add_argument('--output_dim', type=int, default=6, help='Number of output features')
    parser.add_argument('--n_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--K', type=int, default=3, help='Number of conv filter taps')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--data-dir', type=str, default='./data/', help='Path to data directory')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--case', type=str, default='14', help='Grid case')
    parser.add_argument('--wandb', default=False, help='Enable wandb logging',action=argparse.BooleanOptionalAction)
    parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction)
    
    # Step 0: Parse arguments in .json if specified 
    #   Step 0.1 Check if .json file is specified
    #   Step 0.2 Parse whatever is in .json file
    args, left_argv = config_parser.parse_known_args() # if passed args BESIDES defined in cfg_parser, store in left_argv
    if args.cfg_json is not None:
        with open(args.cfg_json) as f:
            json_dict = json.load(f)
        # args.__dict__.update(json_dict) # does not guarantee arg format is correct
        json_argv = []
        for key, value in json_dict.items():
            json_argv.append('--' + key)
            json_argv.append(str(value))
        parser.parse_known_args(json_argv, args)
    
    # Step 1: Parse arguments in command line and override .json values 
    parser.parse_args(left_argv, args) # override JSON values with command-line values

    
    return args