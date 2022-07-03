# The main flow should be here
import argparse
from experiments.rnn import rnn_evaluate, train
from experiments.bert import ner_bert

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="train", required=True)
    parser.add_argument("--module", "-b", default="bert", required=True)
    parser.add_argument("--save", "-s", default="False", required=False)
    args = vars(parser.parse_args())

    pipeline(args)



def pipeline(args):

    if args['mode'] == 'train' and args['module'] == 'rnn':
        train._train(args)

    elif args['mode'] == 'evaluate' and args['module'] == 'rnn':
        rnn_evaluate.plot_results()

    elif args['mode'] == 'train' and args['module'] == 'bert':
        ner_bert._train()



if __name__ == '__main__':
    run()










