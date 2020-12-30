import torch
import numpy as np
from os import path
from NetworksStructure import *
from Procedures import *
from DecodeandEval import *



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LR = 0.000001
    maxepoch = 25
    Number_of_Iterations = 4
    Number_of_steps = 3  # 1.Train only Autoencoders. 2.Train Autoencoders and Butterfyl path 3.Train all three paths
    Print_Interval = 1000
    Retrain = True

    # First Step: Finding embeddings by using Sentence BERT
    # https://github.com/aneesha/SiameseBERT-Notebook/blob/master/SiameseBERT_SemanticSearch.ipynb
    if (path.exists('./embeddings/embpos200k.npy') == False) or (path.exists('./embeddings/embneg200k.npy') == False):
        Obtaining_Embeddings()

    # Second Step: Using embeddings from sentence BERT in order to train three different paths for style transformation
    if (path.exists('./models/posdecoder') == False) or (Retrain == True):
        Embeddings_Pos = np.load('./embeddings/embpos200k.npy')
        Embeddings_Neg = np.load('./embeddings/embneg200k.npy')
        Training_procedure(LR, maxepoch, Number_of_Iterations, Number_of_steps, Print_Interval, device, Embeddings_Pos,
                           Embeddings_Neg)

    # Third Step: Using BERT masked LM head to transfer embeddings into sentences and evaluating performance
    # Got help from: https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb(to use BERT as a generative model)
    # and https://github.com/fastnlp/style-transformer(only evaluation part)
    DecandEval('./dataset/testneg.csv', './dataset/negref.txt', 'Neg to Pos') # Evaluation of transforming negative to positive sentences
    DecandEval('./dataset/testpos.csv', './dataset/posref.txt', 'Pos to neg') # Evaluation of transforming positive to negative sentences
