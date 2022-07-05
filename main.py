"""
Main script to control the activated module and in which mode
4 options is available:
a. training lstm network
b. evaluating lstm network
c. fine-tune pre-trained bert architecture
d. evaluate fine-tuned bert
"""
import argparse
from experiments.rnn import rnn_evaluate, train
from experiments.bert import ner_bert, bert_evaluate

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

    elif args['mode'] == 'evaluate' and args['module'] == 'bert':
        bert_evaluate.evaluate()



if __name__ == '__main__':
    run()










