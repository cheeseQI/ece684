import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def gen(tokens):
    print(tokens)
    indexes = tokenizer.convert_tokens_to_ids(tokens)
    print(indexes)

def main():
    gen(['i', 'love', 'the', 'united kingdom'])

if __name__ == "__main__":
    main()