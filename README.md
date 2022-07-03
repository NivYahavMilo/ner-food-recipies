# NLP Course Final Project Template

## Repo intro

This repository implements a 2 new architectures for "TASTEset" dataset
We provide 2 experiments modules

### 1. Rnn-based approach:

     Embedding layer (64 sized vector)
     lstm layer (32 hidden units)
     fully-connected layer (labels)
  
### 2. Transformer-based approach (fine-tune pretrained model):

      Embeddings - 'bert-base-uncased'
      SequenceTagger
      
## Installation | Requirements
This project was tested on python 3.9 torch &&
for installing the dependency packeges simply:
`pip install -r requirements.txt`

## Quickstart
#### run RNN module:
for training run: (if you wish to save the model set the --save flag to True)
`python main.py --mode train --module rnn` --save False
for evaluation run:
`python main.py --mode evaluate --module rnn`
#### run bert module:
for training run:
`python main.py --mode train --module bert`
for evaluation run:
`python main.py --mode evaluate --module bert`

## Resources
* Paperwithcode link: https://paperswithcode.com/paper/tasteset-recipe-dataset-and-food-entities
* flair frame work: https://github.com/flairNLP/flair

