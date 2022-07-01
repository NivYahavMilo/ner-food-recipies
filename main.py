# The main flow should be here
import argparse


def run():

    args = argparse.ArgumentParser()
    args.add_argument("--mode", "-m", default="inference", required=True)
    args.add_argument("--module", "-b", default="bert", required=True)
    args.add_argument("--input", "-i", default="path", required=True)



