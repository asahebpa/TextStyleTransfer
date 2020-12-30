from os import path
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re
from random import shuffle
import random
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import math
import time
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from EncDecStructure import *

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
model_version = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_version, output_hidden_states=False)
model = BertForMaskedLM.from_pretrained(model_version, config=config)
model.train()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()

tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
model2 = SentenceTransformer('bert-base-nli-mean-tokens')
model2.eval()

def DecandEval(PathtoData, PathtoRef, mode):
    df = pd.read_csv(PathtoData)
    TextData = df.text
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    humanref = list(open(PathtoRef, "r"))
    EncoderNet_Neg = EncoderNet()
    if path.exists('./models/negencoder'):
        checkpoint = torch.load('./models/negencoder')
        EncoderNet_Neg.load_state_dict(checkpoint)
    EncoderNet_Neg.to(device)
    EncoderNet_Neg.eval()
    DecoderNet_Neg = DecoderNet()
    if path.exists('./models/negdecoder'):
        checkpoint = torch.load('./models/negdecoder')
        DecoderNet_Neg.load_state_dict(checkpoint)
    DecoderNet_Neg.to(device)
    DecoderNet_Neg.eval()
    EncoderNet_Pos = EncoderNet()
    if path.exists('./models/posencoder'):
        checkpoint = torch.load('./models/posencoder')
        EncoderNet_Pos.load_state_dict(checkpoint)
    EncoderNet_Pos.to(device)
    EncoderNet_Pos.eval()
    DecoderNet_Pos = DecoderNet()
    if path.exists('./models/posdecoder'):
        checkpoint = torch.load('./models/posdecoder')
        DecoderNet_Pos.load_state_dict(checkpoint)
    DecoderNet_Pos.to(device)
    DecoderNet_Pos.eval()

    n_samples = 1
    batch_size = 1
    top_k = 13
    temperature = 1.0
    generation_mode = "my-sequential"
    burnin = 250
    sample = True
    sumiself = 0
    sumichange = 0
    cc = 0
    refcount = 0

    for cici in TextData:
        sentinit = "[CLS] " + cici
        entinitspli = sentinit.split()
        max_len = len(entinitspli) - 1
        max_iter = int(np.floor(max_len / 1))  # max_len
        leed_out_len = max_len  # max_len
        senttoseed = " ".join(entinitspli[:])
        senfirstiever = " ".join(entinitspli[1:])
        Embd = np.asarray(model2.encode([senfirstiever]))
        Embd = torch.FloatTensor(Embd).to(device)
        if mode == 'Neg to Pos':
            EncoderNet_Neg_Output = EncoderNet_Neg(Embd)
            DecoderNet_Pos_Output = DecoderNet_Pos(EncoderNet_Neg_Output)
            Embd = torch.detach(DecoderNet_Pos_Output)
        if mode == 'Pos to Neg':
            EncoderNet_Pos_Output = EncoderNet_Pos(Embd)
            DecoderNet_Neg_Output = DecoderNet_Neg(EncoderNet_Pos_Output)
            Embd = torch.detach(DecoderNet_Neg_Output)
        seed_text = senttoseed.split()
        optimizer = optim.AdamW(model.parameters(), lr=0.000001)
        lossi = []
        texti = []
        model.train()
        for epoch in range(100): #was range(100)
            bert_sents = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                                  generation_mode=generation_mode,
                                  sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                                  cuda=cuda)
            candid = []
            for sent in bert_sents:
                candid.append(" ".join(sent[:]))
            Embd2 = torch.FloatTensor(model2.encode(candid)).cuda()
            loss = torch.nn.MSELoss(reduction='mean')
            Embd2.requires_grad = True
            output = loss(Embd2, Embd)
            lossi.append(output.item())
            texti.append(candid[0])
            optimizer.zero_grad()
            output.backward(retain_graph=True)
            optimizer.step()
        idc = np.argmin(np.asarray(lossi))
        print('result:')
        refi = "".join(humanref[refcount]);
        refi = '[CLS] '+refi[:-1]+' [SEP]'
        print(refi[6:-6])
        resi = texti[idc]
        print(resi[6:-6])

        bleumleuself = nltk_bleu(senttoseed[6:], resi[6:-6])
        bleumleuchange = nltk_bleu(refi[6:-6], resi[6:-6])
        sumiself += bleumleuself
        sumichange += bleumleuchange
        cc = cc + 1
        refcount = refcount + 1
        print('blues self till now: ' + str(sumiself/cc))
        print('blues change till now: ' + str(sumichange / cc))
        print('.............................')
    print('blues self till now: ' + str(sumiself / cc))
    print('blues change till now: ' + str(sumichange / cc))


def nltk_bleu(texts_origin, text_transfered):
    texts_origin = word_tokenize(texts_origin.lower().strip())
    text_transfered = word_tokenize(text_transfered.lower().strip())
    texts = []
    texts.append(texts_origin)
    return sentence_bleu(texts, text_transfered) * 100
def read_first():
    df = pd.read_csv('dataset/trainneg.csv')
    textneg = np.asarray(df.text)
    df = pd.read_csv('dataset/trainpos.csv')
    TextData = np.asarray(df.text)
    text = np.hstack((textneg, TextData))
    text = text
    MIN = 30
    STOPWORDS = True
    pre1 = []
    for i in text:
        clean = clean_sen_trans(i)
        if len(clean) > MIN:
            pre1.append(clean)
    pre1 = np.asarray(pre1)
    np.save('./dataset/textall.npy', pre1)

def clean_sen_trans(sent):
    clean_list = ''
    for word in sent.split():
        # print(word)
        if word not in stop_words:
            word = re.sub('[\W_]+', '', word)
            word = re.sub(' +', ' ', word)
            clean_list += word.lower() + str(' ')
    return clean_list
def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * 0 + [SEP] for _ in range(batch_size)]
    # if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))
def my_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                                   burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        kk = np.random.randint(0, seed_len-1)+1
        # kk = ii % max_len
        for jj in range(batch_size):
            batch[jj][kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        out = out[0]
        # topk = top_k if (ii >= burnin) else 0
        # idxs = generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        idxs = generate_step(out, gen_idx=kk, top_k= top_k, temperature=None, sample=False)
        for jj in range(batch_size):
            batch[jj][kk] = idxs[jj]
        for_print = tokenizer.convert_ids_to_tokens(batch[0])
        for_print = for_print[:kk + 1] + ['(*)'] + for_print[kk + 1:]
    return untokenize_batch(batch)


def parallel_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                                   burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        np.random.seed(13)
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        out = out[0]
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len + kk, top_k=0, temperature=temperature, sample=False)
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = idxs[jj]
        for_print = tokenizer.convert_ids_to_tokens(batch[0])
        for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]


    return untokenize_batch(batch)


def parallel_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True,
                        cuda=False, print_every=10, verbose=True):
    """ Generate for all positions at each time step """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        for kk in range(max_len):
            idxs = generate_step(out, gen_idx=seed_len + kk, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = idxs[jj]


    return untokenize_batch(batch)


def sequential_generation(seed_text, batch_size=10, max_len=15, leed_out_len=15,
                          top_k=0, temperature=None, sample=True, cuda=False):
    """ Generate one word at a time, in L->R order """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_len):
        inp = [sent[:seed_len + ii + leed_out_len] + [sep_id] for sent in batch]
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        # idxs = generate_step(out, gen_idx=seed_len + ii, top_k=top_k, temperature=temperature, sample=sample)
        idxs = generate_step(out, gen_idx=seed_len + ii, top_k=0, temperature=temperature, sample=True)
        for jj in range(batch_size):
            batch[jj][seed_len + ii] = idxs[jj]

    return untokenize_batch(batch)


def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25,
             generation_mode="parallel-sequential",
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1, leed_out_len=0):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                   cuda=cuda, verbose=False)
        elif generation_mode == "sequential":
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                          temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                          cuda=cuda)
        elif generation_mode == "parallel":
            batch = parallel_generation(seed_text, batch_size=batch_size,
                                        max_len=max_len, top_k=top_k, temperature=temperature,
                                        sample=sample, max_iter=max_iter,
                                        cuda=cuda, verbose=False)
        elif generation_mode == "my-sequential":
            batch = my_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                   cuda=cuda, verbose=False)

        if (batch_n + 1) % print_every == 0:
            start_time = time.time()

        sentences += batch
    return sentences

