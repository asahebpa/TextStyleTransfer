import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os
from EncDecStructure import *
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords


def Obtaining_Embeddings():
    SentenceBertEnc = SentenceTransformer('bert-base-nli-mean-tokens')
    SentenceBertEnc.eval()
    print('ggg')
    if path.exists('./embeddings') == False:
        os.mkdir('./embeddings')
    Positive = pd.read_csv('./dataset/trainpos.csv')
    Negative = pd.read_csv('./dataset/trainneg.csv')
    TextPos = Positive.text[:200000]
    Embeddings_Pos = np.array(SentenceBertEnc.encode(TextPos))
    TextNeg = Positive.text[:200000]
    Embeddings_Neg = np.array(SentenceBertEnc.encode(TextNeg))
    np.save('./embeddings/embpos200k.npy', Embeddings_Pos)
    np.save('./embeddings/embneg200k.npy', Embeddings_Neg)
    del SentenceBertEnc
    del Embeddings_Pos
    del Embeddings_Neg



def Training_procedure(LR, maxepoch, Number_of_Iterations, Number_of_steps, Print_Interval, device, datapos, dataneg):
    if path.exists('./models') == False:
        os.mkdir('./models')
    Embeddings_Pos_Mean = torch.FloatTensor((np.mean(datapos, axis=0).reshape(-1, 1)).T).to(device)
    Embeddings_Neg_Mean = torch.FloatTensor((np.mean(dataneg, axis=0).reshape(-1, 1)).T).to(device)
    trainloaderpos = torch.utils.data.DataLoader(datapos, batch_size=320,
                                              shuffle=True, num_workers=20)
    trainloaderneg = torch.utils.data.DataLoader(dataneg, batch_size=320,
                                                 shuffle=True, num_workers=20)

    for Iteration in range(Number_of_Iterations):
        for Step in range(Number_of_steps):
            if Step == 0:
                print('Step 1 started.')
                ##### Training Autoencoder Path for Negative Sentences
                print('Training Autoencoder path for Negative Sentences.')
                EncoderNet_Neg = EncoderNet()
                if path.exists('./models/negencoder'):
                    checkpoint = torch.load('./models/negencoder')
                    EncoderNet_Neg.load_state_dict(checkpoint)
                EncoderNet_Neg.to(device)
                EncoderNet_Neg.train()
                DecoderNet_Neg = DecoderNet()
                if path.exists('./models/negdecoder'):
                    checkpoint = torch.load('./models/negdecoder')
                    DecoderNet_Neg.load_state_dict(checkpoint)
                DecoderNet_Neg.to(device)
                DecoderNet_Neg.train()
                params = list(EncoderNet_Neg.parameters()) + list(DecoderNet_Neg.parameters())
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(params, lr=LR)
                for epoch in range(maxepoch):
                    running_loss = 0.0
                    print('Epoch: ', epoch)
                    for i, data in enumerate(trainloaderneg, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        optimizer.zero_grad()
                        EncoderNet_Neg_Output = EncoderNet_Neg(Inputs)
                        outputs = DecoderNet_Neg(EncoderNet_Neg_Output)
                        loss = criterion(outputs, Targets)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Negative Sentences Autoencoder: %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0
                torch.save(EncoderNet_Neg.state_dict(), './models/negencoder')
                torch.save(DecoderNet_Neg.state_dict(), './models/negdecoder')
                del EncoderNet_Neg
                del DecoderNet_Neg
                del optimizer
                ##### Training Autoencoder Path for Positive Sentences
                print('Training Autoencoder path for Positive Sentences.')
                EncoderNet_Pos = EncoderNet()
                if path.exists('./models/posencoder'):
                    checkpoint = torch.load('./models/posencoder')
                    EncoderNet_Pos.load_state_dict(checkpoint)
                EncoderNet_Pos.to(device)
                EncoderNet_Pos.train()
                DecoderNet_Pos = DecoderNet()
                if path.exists('./models/posdecoder'):
                    checkpoint = torch.load('./models/posdecoder')
                    DecoderNet_Pos.load_state_dict(checkpoint)
                DecoderNet_Pos.to(device)
                DecoderNet_Pos.train()
                params = list(EncoderNet_Pos.parameters()) + list(DecoderNet_Pos.parameters())
                optimizer = optim.AdamW(params, lr=LR)
                for epoch in range(maxepoch):
                    running_loss = 0.0
                    print('Epoch: ', epoch)
                    for i, data in enumerate(trainloaderpos, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        optimizer.zero_grad()
                        EncoderNet_Pos_Output = EncoderNet_Pos(Inputs)
                        outputs = DecoderNet_Pos(EncoderNet_Pos_Output)
                        loss = criterion(outputs, Targets)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Positive Sentences Autoencoder: %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0
                torch.save(EncoderNet_Pos.state_dict(), './models/posencoder')
                torch.save(DecoderNet_Pos.state_dict(), './models/posdecoder')
                del EncoderNet_Pos
                del DecoderNet_Pos
                del optimizer
                print('Step 1 finished.')

            ##### Butterfly Path(Unsupervised)
            if Step == 1:
                print('Step 2 started.')
                print('Training Unsupervised(Butterfly) Path.')
                EncoderNet_Neg = EncoderNet()
                if path.exists('./models/negencoder'):
                    checkpoint = torch.load('./models/negencoder')
                    EncoderNet_Neg.load_state_dict(checkpoint)
                EncoderNet_Neg.to(device)
                EncoderNet_Neg.train()
                DecoderNet_Neg = DecoderNet()
                if path.exists('./models/negdecoder'):
                    checkpoint = torch.load('./models/negdecoder')
                    DecoderNet_Neg.load_state_dict(checkpoint)
                DecoderNet_Neg.to(device)
                DecoderNet_Neg.train()
                EncoderNet_Pos = EncoderNet()
                if path.exists('./models/posencoder'):
                    checkpoint = torch.load('./models/posencoder')
                    EncoderNet_Pos.load_state_dict(checkpoint)
                EncoderNet_Pos.to(device)
                EncoderNet_Pos.train()
                DecoderNet_Pos = DecoderNet()
                if path.exists('./models/posdecoder'):
                    checkpoint = torch.load('./models/posdecoder')
                    DecoderNet_Pos.load_state_dict(checkpoint)
                DecoderNet_Pos.to(device)
                DecoderNet_Pos.train()
                params = list(EncoderNet_Neg.parameters()) + list(DecoderNet_Neg.parameters()) + \
                         list(EncoderNet_Pos.parameters()) + list(DecoderNet_Pos.parameters())
                optimizer = optim.AdamW(params, lr=LR)
                for epoch in range(maxepoch):
                    print('Epoch: ', epoch)
                    ##### Butterfly paths starting from positive sentences
                    for i, data in enumerate(trainloaderpos, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        optimizer.zero_grad()
                        EncoderNet_Pos_Output = EncoderNet_Pos(Inputs)
                        DecoderNet_Neg_Output = DecoderNet_Neg(EncoderNet_Pos_Output)
                        EncoderNet_Neg_Output = EncoderNet_Neg(DecoderNet_Neg_Output)
                        DecoderNet_Pos_Output = DecoderNet_Pos(EncoderNet_Neg_Output)
                        Self_Neg = DecoderNet_Neg(EncoderNet_Neg_Output)
                        Self_Pos = DecoderNet_Pos(EncoderNet_Pos_Output)
                        loss1 = criterion(DecoderNet_Pos_Output, Targets) # Total butterfly path loss
                        loss2 = criterion(Self_Neg, DecoderNet_Neg_Output) # Lower half self loss
                        loss3 = criterion(Self_Pos, Targets)  # Upper half self loss
                        loss = loss1 + loss2 + loss3
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Butterfly Path(starting from positive sentences): %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0

                    ##### Butterfly paths starting from negative sentences
                    for i, data in enumerate(trainloaderneg, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        optimizer.zero_grad()
                        EncoderNet_Neg_Output = EncoderNet_Neg(Inputs)
                        DecoderNet_Pos_Output = DecoderNet_Pos(EncoderNet_Neg_Output)
                        EncoderNet_Pos_Output = EncoderNet_Pos(DecoderNet_Pos_Output)
                        DecoderNet_Neg_Output = DecoderNet_Neg(EncoderNet_Pos_Output)
                        Self_Neg = DecoderNet_Neg(EncoderNet_Neg_Output)
                        Self_Pos = DecoderNet_Pos(EncoderNet_Pos_Output)
                        loss1 = criterion(DecoderNet_Neg_Output, Targets) # Total butterfly path loss
                        loss2 = criterion(Self_Neg, Targets) # Lower half self loss
                        loss3 = criterion(Self_Pos, DecoderNet_Pos_Output) # Upper half self loss
                        loss = loss1 + loss2 + loss3
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Butterfly Path(starting from negative sentences): %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0
                torch.save(EncoderNet_Neg.state_dict(), './models/negencoder')
                torch.save(DecoderNet_Neg.state_dict(), './models/negdecoder')
                torch.save(EncoderNet_Pos.state_dict(), './models/posencoder')
                torch.save(DecoderNet_Pos.state_dict(), './models/posdecoder')
                del EncoderNet_Neg
                del DecoderNet_Neg
                del EncoderNet_Pos
                del DecoderNet_Pos
                del optimizer
                print('Step 2 finished.')
            ###### Semi-supervised Path
            if Step == 2:
                print('Step 3 started.')
                print('Training Semi-supervised Path.')
                EncoderNet_Neg = EncoderNet()
                if path.exists('./models/negencoder'):
                    checkpoint = torch.load('./models/negencoder')
                    EncoderNet_Neg.load_state_dict(checkpoint)
                EncoderNet_Neg.to(device)
                EncoderNet_Neg.train()
                DecoderNet_Neg = DecoderNet()
                if path.exists('./models/negdecoder'):
                    checkpoint = torch.load('./models/negdecoder')
                    DecoderNet_Neg.load_state_dict(checkpoint)
                DecoderNet_Neg.to(device)
                DecoderNet_Neg.train()
                EncoderNet_Pos = EncoderNet()
                if path.exists('./models/posencoder'):
                    checkpoint = torch.load('./models/posencoder')
                    EncoderNet_Pos.load_state_dict(checkpoint)
                EncoderNet_Pos.to(device)
                EncoderNet_Pos.train()
                DecoderNet_Pos = DecoderNet()
                if path.exists('./models/posdecoder'):
                    checkpoint = torch.load('./models/posdecoder')
                    DecoderNet_Pos.load_state_dict(checkpoint)
                DecoderNet_Pos.to(device)
                DecoderNet_Pos.train()
                params = list(EncoderNet_Neg.parameters()) + list(DecoderNet_Neg.parameters()) +\
                         list(EncoderNet_Pos.parameters()) + list(DecoderNet_Pos.parameters())
                optimizer = optim.AdamW(params, lr=LR)
                for epoch in range(maxepoch):
                    running_loss = 0.0
                    print('Epoch: ', epoch)
                    ##### Semi-supervised paths starting from positive sentences
                    for i, data in enumerate(trainloaderpos, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        Targets = torch.detach(Targets)
                        EncoderNet_Pos_Output = EncoderNet_Pos(Inputs)
                        DecoderNet_Neg_Output = DecoderNet_Neg(EncoderNet_Pos_Output)
                        # Constructing corresponding negative sentence by adjusting mean
                        torch.detach(Embeddings_Pos_Mean)
                        torch.detach(Embeddings_Neg_Mean)
                        Target_SemiSup = Targets - 3 * (Embeddings_Pos_Mean - Embeddings_Neg_Mean)
                        Target_SemiSup = torch.detach(Target_SemiSup)
                        loss = criterion(DecoderNet_Neg_Output, Target_SemiSup)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Semi-supervised Path(starting from positive sentences): %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0
                    ##### Semi-supervised paths starting from negative sentences
                    for i, data in enumerate(trainloaderneg, 0):
                        Inputs = data.to(device)
                        Targets = data.to(device)
                        Targets = torch.detach(Targets)
                        optimizer.zero_grad()
                        EncoderNet_Neg_Output = EncoderNet_Neg(Inputs)
                        DecoderNet_Pos_Output = DecoderNet_Pos(EncoderNet_Neg_Output)
                        # Constructing corresponding positive sentence by adjusting mean
                        torch.detach(Embeddings_Pos_Mean)
                        torch.detach(Embeddings_Neg_Mean)
                        Target_SemiSup = Inputs - Embeddings_Neg_Mean + 2 * Embeddings_Pos_Mean
                        Target_SemiSup = torch.detach(Target_SemiSup)
                        loss = criterion(DecoderNet_Pos_Output, Target_SemiSup)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % Print_Interval == 0:
                            print('[%d, %5d] Loss of Semi-supervised Path(starting from negative sentences): %.9f' %
                                  (epoch + 1, i + 1, running_loss / Print_Interval))
                            running_loss = 0.0
                torch.save(EncoderNet_Neg.state_dict(), './models/negencoder')
                torch.save(DecoderNet_Neg.state_dict(), './models/negdecoder')
                torch.save(EncoderNet_Pos.state_dict(), './models/posencoder')
                torch.save(DecoderNet_Pos.state_dict(), './models/posdecoder')
                del EncoderNet_Neg
                del DecoderNet_Neg
                del EncoderNet_Pos
                del DecoderNet_Pos
                del optimizer
                print('Step 3 finished.')

