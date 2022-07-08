# NLP Course Final Project

## Repo intro

This repository implements a 2 new architectures for "TASTEset" dataset
We provide 2 experiments modules

### 1. Rnn-based approach:

     Embedding layer
     lstm layer
     fully-connected layer
  
### 2. Transformer-based approach (fine-tune pretrained model):

      Embeddings - 'bert-base-uncased'
      SequenceTagger - 'ner'
      
## Installation | Requirements
This project was tested on python 3.9 & torch 1.12.0 & flair 0.11.3 
for installing the dependency packeges simply:

it is recmmended to create virtual env inside this repo.
in cmd:
`py -m venv env
cd env
Scripts\activate`

and then:

`pip install -r requirements.txt`

## Quickstart
#### Run RNN module:
for training run:

`python main.py --mode train --module rnn --save False`

(if you wish to save the model set the --save flag to True)

for evaluation run:

`python main.py --mode evaluate --module rnn`

#### Run BERT module:

for training run:

`python main.py --mode train --module bert`

for evaluation run:

`python main.py --mode evaluate --module bert`

This project contains the pt model in the repository. (uploaded using GIT LFS)

##### Data Utils
converting raw data to IOB tagging format:

`python main.py --mode IOB`

note: the scripts saves it to csv file by the name "IOB tagging.csv"

converting the "IOB tagging.csv" to flair input (train.txt, dev.txt, test.txt):

`python main.py --mode generate_corpus`


## Resources
* Paperwithcode link: https://paperswithcode.com/paper/tasteset-recipe-dataset-and-food-entities
* flair frame work: https://github.com/flairNLP/flair

