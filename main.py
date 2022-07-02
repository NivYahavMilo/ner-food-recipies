# The main flow should be here
import argparse


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="inference", required=True)
    parser.add_argument("--module", "-b", default="bert", required=True)
    parser.add_argument("--input", "-i", default="path", required=True)
    args = vars(parser.parse_args())



