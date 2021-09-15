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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader , Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from preprocessor import *

def progressLearning(value, endvalue, loss, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f}".format(arrow + spaces, value+1, endvalue, loss))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def weight_fn(occ_output, device) :
    x_max = torch.tensor(100.0, dtype = torch.float32, device = device)
    occ_weight = torch.where(occ_output > x_max, 
        (occ_output/x_max), 
        torch.tensor(1.0, dtype=torch.float32, device = device)
    )
    occ_weight = torch.pow(occ_weight, (3/4))
    return occ_weight

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Text Data
    text_data = pd.read_csv(args.data_dir)
    text_list = list(text_data['번역문'])

    # -- Tokenize & Encoder
    en_processor = Preprocessor(text_list, en_preprocess, word_tokenize)

    bpe_code_path = os.path.join(args.token_dir, 'bpe_code.csv')
    subword_data_path = os.path.join(args.token_dir, 'subword_data.csv')

    if os.path.exists(bpe_code_path) and os.path.exists(subword_data_path) :
        print('Load Binary Pair Encoding')
        bpe_code = pd.read_csv(bpe_code_path)
        subword_list = pd.read_csv(subword_data_path)
        en_tokenizer = Tokenizer(bpe_code, en_preprocess, word_tokenize)
        en_encoder = Encoder(subword_list)
    else :
        print('Start Binary Pair Encoding')
        bpe_code, subword_list = en_processor.get_bpe(args.merge_count)
        en_tokenizer = Tokenizer(bpe_code, en_preprocess, word_tokenize)
        en_encoder = Encoder(subword_list)

        print('Save Binary Pair Encoding')
        bpe_data = en_tokenizer.get_data() # bpe code data
        bpe_df = pd.DataFrame({'data' : list(bpe_data.keys()) , 'count' : list(bpe_data.values())})
        bpe_df.to_csv(bpe_code_path)
        subword_data = en_encoder.get_data() # subword token data
        subword_df = pd.DataFrame({'token' : list(subword_data.keys()) , 'index' : list(subword_data.values())})
        subword_df.to_csv(subword_data_path)

    en_token = en_encoder.get_data()
    token_size = len(en_token)

    # -- Encoding & Making Index Data
    idx_data = []
    for sen in text_list :
        tok_list = en_tokenizer.tokenize(sen)
        idx_list = [Token.SOS] + en_encoder.encode(tok_list) + [Token.EOS]
        idx_data.append(idx_list)

    # -- Dataset
    print('Making Model Dataset')
    ngram_dset = NgramDataset(token_size, args.window_size)
    con_data, tar_data, occ_data = ngram_dset.get_data(idx_data)
    glove_dset = GloveDataset(con_data, tar_data, occ_data, args.val_ratio)
    train_dset, val_dset = glove_dset.split()

    # -- DataLoader
    train_loader = DataLoader(train_dset,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(val_dset,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        batch_size=args.val_batch_size
    )

    # -- Model
    model = Glove(args.embedding_size, token_size).to(device)

    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # -- Loss
    criterion = nn.MSELoss()

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Training
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    for epoch in range(args.epochs) :
        idx = 0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        for data in train_loader :
            con_data = data['con'].long().to(device)
            tar_data = data['tar'].long().to(device)

            occ_data = data['occ'].float().to(device)
            occ_weight = weight_fn(occ_data, device)
            occ_log = torch.log(occ_data)
            occ_label = torch.mul(occ_weight, occ_log)

            occ_output = model(con_data, tar_data)

            loss = criterion(occ_output, occ_label)
            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item())

            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                log_count += 1
            idx += 1

        with torch.no_grad() :
            model.eval()
            val_loss = 0.0
            for data in val_loader :
                con_data = data['con'].long().to(device)
                tar_data = data['tar'].long().to(device)

                occ_data = data['occ'].float().to(device)
                occ_weight = weight_fn(occ_data, device)
                occ_log = torch.log(occ_data)
                occ_label = torch.mul(occ_weight, occ_log)
                
                occ_output = model(con_data, tar_data)

                loss = criterion(occ_output, occ_label)
                val_loss += loss

            val_loss /= len(val_loader)
        writer.add_scalar('val/loss', val_loss.item(), epoch)

        if val_loss < min_loss :
            min_loss = val_loss
            torch.save({'epoch' : (epoch) ,  
                'model_state_dict' : model.state_dict() , 
                'loss' : val_loss.item()}, 
            os.path.join(args.model_dir,'checkpoint_glove.pt'))
            stop_count = 0
        else :
            stop_count += 1
            if stop_count >= 5 :
                print('\nTraining Early Stopped') 
                break

        scheduler.step()
        print('\nVal Loss : %.3f \n' %val_loss)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--merge_count', type=int, default=6000, help='number of bpe merge (default: 6000)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--window_size', type=int, default=13, help='window size (default: 13)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='input batch size for validing (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 2e-4)')    
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.1)')

    parser.add_argument('--data_dir', type=str, default='../Data/korean_dialogue_translation.csv', help = 'text data')
    parser.add_argument('--token_dir', type=str, default='./Token/english' , help='token data dir path')
    parser.add_argument('--embedding_dir', type=str, default='./Embedding/english' , help='embedding dir path')
    parser.add_argument('--model_dir', type=str, default='./Model/english' , help='best model dir path')
    parser.add_argument('--log_dir' , type=str , default='./Log/english', help = 'log data dir path')

    args = parser.parse_args()

    train(args)
