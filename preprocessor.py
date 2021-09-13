import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import collections
import random
import re

def kor_preprocess(sen) :
    sen = re.sub('[0-9]+,*[0-9]*', 'NUM ', sen)
    sen = re.sub('[^A-Z가-힣!?.,\']', ' ' , sen)
    sen = re.sub(' {2,}', ' ', sen)
    return sen

def en_preprocess(sen) :
    sen = sen.lower()
    re.sub('[0-9]+,*[0-9]*', 'NUM ', sen)
    sen = re.sub('[^a-z!?.,\']' , ' ' , sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen

class Preprocessor :
    def __init__(self, text, tokenize, preprocess) :
        self.data = [tokenize(preprocess(sen)) for sen in tqdm(text)]
        self.preprocess = preprocess
        self.tokenize = tokenize

    def build_dict(self) :
        counter = collections.Counter()
        for tok_list in self.data :
            counter.update(tok_list)
        counter = dict(counter)
        
        bpe_dict = {}
        for tok, counts in counter.items() :
            ch_list = tuple(tok) + ('</w>',)
            ch_str = ' '.join(ch_list)
            bpe_dict[ch_str] = counts

        return bpe_dict

    def get_stats(self, bpe_dict):
        pairs = collections.defaultdict(int)
        for word, count in bpe_dict.items():
            subword_list = word.split()
            subword_size = len(subword_list)
            for i in range(subword_size-1):
                pairs[subword_list[i], subword_list[i+1]] += count
        return pairs

    def merge_dict(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    def get_bpe(self, merge_count) :
        bpe_dict = self.build_dict()
        bpe_codes = {}
        
        for i in tqdm(range(merge_count)) :
            pairs = self.get_stats(bpe_dict)
            best = max(pairs, key=pairs.get)
            bpe_dict = self.merge_dict(best, bpe_dict)
            bpe_codes[best] = i
            
        return bpe_codes , bpe_dict
       
class Tokenizer :
    def __init__(self, bpe_code, tokenize , preprocess) :
        self.bpe_code = bpe_code
        self.preprocess = preprocess
        self.tokenize_fn = tokenize
    
    def get_pairs(self, word) :
        pairs = set()
        prev_char = word[0]
        for char in word[1:] :
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def encode_word(self, orig) :
        word = tuple(orig) + ('</w>',)
        pairs = self.get_pairs(word)    

        if not pairs:
            return orig

        iteration = 0
        while True:
            iteration += 1
            bigram = min(pairs, key = lambda pair: self.bpe_code.get(pair, float('inf')))
            if bigram not in self.bpe_code:
                break
            first, second = bigram # first tok, second tok
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>',''),)

        return word

    def tokenize(self, sen) :
        sen = self.preprocess(sen)
        tok_list = self.tokenize_fn(sen)
        subword_list = []
        for tok in tok_list :
            subwords = self.encode_word(tok)
            subword_list += list(subwords)
            
        return subword_list


class Convertor :
    def __init__(self, bpe_vocab) :
        self.bpe_vocab = bpe_vocab
        self.sub2idx, self.idx2sub = self.build_dict()
    
    def build_dict(self) :
        subword_list = []
        for tok in self.bpe_vocab.keys() :
            tok = tok[:-4]
            subwords = tok.split(' ')
            subword_list += [sub for sub in subwords if sub != '']
            
        random.shuffle(subword_list)
        subword_list = list(set(subword_list))
        
        sub2idx = dict(zip(subword_list, range(1,len(subword_list)+1)))
        idx2sub = dict(zip(range(1,len(subword_list)+1), subword_list))
    
        return sub2idx, idx2sub

    def set_sub2idx(self, sub2idx) :
        self.sub2idx = sub2idx

    def set_idx2sub(self, idx2sub) :
        self.idx2sub = idx2sub

    def encode(self, tok_list) :    
        idx_list = [self.sub2idx[tok] for tok in tok_list if tok != ' ']
        return idx_list              
    
    def decode(self, idx_list) :
        tok_list = [self.idx2sub[idx] for idx in idx_list]
        return tok_list
        
    def get_sub2idx(self) :
        return self.sub2idx

    def get_idx2sub(self) :
        return self.idx2sub
    
    def get_size(self) :
        return len(self.sub2idx)+1


