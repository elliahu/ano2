import netron
import argparse

parser = argparse.ArgumentParser(description="NN visualization")
parser.add_argument("model", help="Classification model to use")
args = parser.parse_args()

netron.start(args.model)