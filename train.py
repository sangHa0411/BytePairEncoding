import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module

from nltk.tokenize import word_tokenize
from konlpy.tag import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader , Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from loader import *
from preprocessor import *

def progressLearning(value, endvalue, loss , acc , bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1 , endvalue , loss , acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def acc_fn(y_output , y_label) :
    y_arg = torch.argmax(y_output, dim=-1)
    y_acc = (y_arg == y_label).float()
    y_acc = torch.mean(y_acc)
    return y_acc

def train(data_dir, model_dir, args) :

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Raw Data
    text_data = pd.read_csv(args.data_dir)
    text_list = list(text_data[args.data_type])

    assert args.data_type in ['원문','번역문']
    if args.data_type == '원문' :
        mecab = Mecab()
        tokenize_fn = mecab.morphs
        preprocess_fn = kor_preprocess
        data_type = 'korean'
    else :
        tokenize_fn = word_tokenize
        preprocess_fn = en_preprocess
        data_type = 'english'

    bpe_code_path = os.path.join('./Data', data_type, 'bpe_code.csv')
    bpe_dict_path = os.path.join('./Data' , data_type, 'bpe_dict.csv')

    if (os.path.exists(bpe_code_path)==True) and (os.path.exists(bpe_dict_path)==True) :
        bpe_code = load_bpe_code(bpe_code_path)
        bpe_dict = load_bpe_dict(bpe_dict_path)
    else :
        bpe_preprocessor = Preprocessor(text_list, tokenize_fn, preprocess_fn)
        bpe_code, bpe_dict = bpe_preprocessor.get_bpe(args.token_size)
        save_bpe_code(bpe_code, bpe_code_path)
        save_bpe_dict(bpe_dict, bpe_dict_path)
        
    # -- Tokenizer & Encoder
    tokenizer = Tokenizer(bpe_code, tokenize_fn, preprocess_fn)
    convertor = Convertor(bpe_dict)

    sub2idx_path = os.path.join('./Data', data_type, 'sub2idx.csv')
    if os.path.exists(sub2idx_path)==True :
        sub2idx, idx2sub = load_sub2idx(sub2idx_path)
        convertor.set_sub2idx(sub2idx)
        convertor.set_idx2sub(idx2sub)
    else :
        sub2idx = convertor.get_sub2idx()
        save_sub2idx(sub2idx, sub2idx_path)

    print('Encoding starts')
    idx_data = []
    for text in tqdm(text_list) :
        tok_list = tokenizer.tokenize(text)
        idx_list = convertor.encode(tok_list)
        idx_data.append(idx_list)
    print('Encoding finished')

    # -- Dataset
    ngram_data = NgramDataset(args.window_size)
    cen_data, neigh_data = ngram_data.get_data(idx_data) 

    dataset = EmbeddingDataset(cen_data, neigh_data)
    train_set, val_set = dataset.split()

    # -- Dataloader
    train_loader = DataLoader(train_set,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(val_set,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        batch_size=args.val_batch_size
    )
    
    # -- Embedding Model
    v_size = convertor.get_size()
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        em_size = args.embedding_size,
        v_size = v_size,
        window_size = args.window_size
    ).to(device)

    # -- Optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # -- Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # -- Logging
    writer = SummaryWriter(os.path.join(args.log_dir, data_type))

    # -- loss
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    print('Training Starts')
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # -- Training
    for epoch in range(args.epochs) :
        print('Epoch : %d' %epoch)
        idx = 0
        for cen_data, neigh_data in train_loader :
            cen_data = cen_data.long().to(device)
            neigh_data = neigh_data.long().to(device)

            neigh_out = model(cen_data)
            neigh_out = torch.reshape(neigh_out, (-1,v_size))
            neigh_data = torch.reshape(neigh_data, (-1,))
            
            loss = loss_fn(neigh_out , neigh_data)
            acc = acc_fn(neigh_out , neigh_data)
        
            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())

            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        with torch.no_grad() :
            model.eval()
            val_loss = 0.0
            val_acc = 0.0

            for cen_data, neigh_data in val_loader :
                cen_data = cen_data.long().to(device)
                neigh_data = neigh_data.long().to(device)

                neigh_out = model(cen_data)        
                neigh_out = torch.reshape(neigh_out, (-1,v_size))
                neigh_data = torch.reshape(neigh_data, (-1,))
            
                val_loss += loss_fn(neigh_out , neigh_data)
                val_acc += acc_fn(neigh_out , neigh_data)

            model.train()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            writer.add_scalar('test/loss', val_loss.item(), epoch)
            writer.add_scalar('test/acc', val_acc.item(), epoch)

        if val_loss < min_loss :
            min_loss = val_loss
            torch.save({'epoch' : (epoch) ,  
                        'model_state_dict' : model.state_dict() , 
                        'loss' : val_loss.item() , 
                        'acc' : val_acc.item()} , 
                        os.path.join(args.model_dir, data_type, 'word2vec_model.pt'))        
            stop_count = 0 
        else :
            stop_count += 1
            if stop_count >= 5 :      
                print('\nTraining Early Stopped')
                break
        scheduler.step()
        print('\nVal Loss : %.3f Val Accuracy : %.3f \n' %(val_loss, val_acc))
    print('Training finished')

    # saving embedding weight
    em_weight = model.embedding.weight
    o_weight = model.o_layer.weight.view(-1,args.window_size-1,args.embedding_size)
    o_weight = torch.mean(o_weight, dim=1)

    em_weight = (em_weight + o_weight)/2
    em_weight = em_weight.detach().cpu().numpy()

    o_bias = model.o_layer.bias.view(-1,args.window_size-1)
    o_bias = torch.mean(o_bias, dim=1)     
    o_bias = o_bias.detach().cpu().numpy()
    
    np.save(os.path.join('./Data',data_type,'em_weight.npy'), em_weight)
    np.save(os.path.join('./Data',data_type,'o_bias.npy'), o_bias)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=77, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 20)')
    parser.add_argument('--token_size', type=int, default=4000, help='number of sub words tokens (default: 5000)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--window_size', type=int, default=9, help='window size (default: 11)')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 512)')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='input batch size for validing (default: 1024)')
    parser.add_argument('--model', type=str, default='SkipGram', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 1e-4)')    
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='../Data/korean_dialogue_translation.csv')
    parser.add_argument('--data_type' , type=str, default='번역문')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
