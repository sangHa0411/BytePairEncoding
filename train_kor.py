import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

from dataset import *
from model import *
from preprocessor import *

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f}, Acc : {4:.3f}".format(arrow + spaces, 
        value+1, 
        endvalue, 
        loss, 
        acc)
    )
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Text Data
    text_data = pd.read_csv(args.data_dir)
    text_list = list(text_data['원문'])

    # -- Tokenize & Encoder
    kor_text_path = os.path.join(args.token_dir, 'korean.txt')
    if os.path.exists(kor_text_path) == False :
        write_data(text_list, kor_text_path, preprocess_kor)
    kor_spm = get_spm(args.token_dir, 'korean.txt', 'kor_spm', args.token_size)

    idx_data = []
    for sen in text_list :
        sen = preprocess_kor(sen)
        idx_list = kor_spm.encode_as_ids(sen)
        idx_data.append(idx_list)

    # -- Dataset
    ngram_dset = NgramDataset(args.token_size, args.window_size)
    cen_data, con_data = ngram_dset.get_data(idx_data)
    word2vec_dset = Word2VecDataset(cen_data, con_data)

    # -- DataLoader
    data_loader = DataLoader(word2vec_dset,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        batch_size=args.batch_size
    )
    
    # -- Model
    model_module = getattr(import_module('model') , args.model) 
    model = model_module(args.embedding_size, args.token_size).to(device)

    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    
    # -- Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    for epoch in range(args.epochs) :
        idx = 0
        loss_train = 0.0
        acc_train = 0.0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        for center, context in data_loader :
            center = center.long().to(device)
            context = context.long().to(device)

            if args.model == 'CBOW' :
                in_data = context
                out_label = center
            else : 
                in_data = center.unsqueeze(1).repeat(1,args.window_size-1)
                in_data = in_data.view([-1,])
                out_label = context.view([-1,])
            
            out_data = model(in_data)

            loss = criterion(out_data, out_label)
            acc = (torch.argmax(out_data,-1) == out_label).float().mean()
            loss.backward()
            optimizer.step()

            loss_train += loss
            acc_train += acc
        
            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            idx += 1

        loss_train /= len(data_loader)
        acc_train /= len(data_loader)

        torch.save({'epoch' : (epoch) ,  
            'model_state_dict' : model.state_dict() , 
            'loss' : loss_train.item(), 
            'acc' : acc_train.item()},
        os.path.join(args.model_dir,'en_'+args.model.lower()+'.pt'))
   
        scheduler.step()
        print('\nMean Loss : %.3f , Mean Acc : %.3f\n' %(loss_train, acc_train))

    kor_weight = model.get_weight()
    kor_weight = kor_weight.detach().cpu().numpy()
    kor_weight[0] = 0.0

    kor_bias = model.get_bias()
    kor_bias = kor_bias.detach().cpu().numpy()
    kor_bias[0] = 0.0

    np.save(os.path.join(args.embedding_dir, 'kor_weight.npy'), kor_weight)
    np.save(os.path.join(args.embedding_dir, 'kor_bias.npy'), kor_bias)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 35)')
    parser.add_argument('--token_size', type=int, default=7000, help='number of bpe merge (default: 7000)')
    parser.add_argument('--model', type=str, default='CBOW', help='model of embedding (default: CBOW)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--window_size', type=int, default=11, help='window size (default: 11)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='input batch size for validing (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')    
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.1)')

    parser.add_argument('--data_dir', type=str, default='../Data/korean_dialogue_translation.csv', help = 'text data')
    parser.add_argument('--token_dir', type=str, default='./Token' , help='token data dir path')
    parser.add_argument('--embedding_dir', type=str, default='./Embedding' , help='embedding dir path')
    parser.add_argument('--model_dir', type=str, default='./Model' , help='best model dir path')

    args = parser.parse_args()

    train(args)
