# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import sys
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertPreTrainedModel #TODO move it

# define parameters
device = 'cuda:0'
n_epochs = 100
runs = 1
patience = 3

lr = 0.001 # learning rate
clip = 5 # Clip the gradient


def main():
    train_loader, dev_loader, test_loader, lang, tokenizer = preprocess_and_load_data()
     
    criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.pad_token_id)
    criterion_intents = nn.CrossEntropyLoss()
    
    #TODO check for multiple runs

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()
    
