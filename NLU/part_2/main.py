# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import sys
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# define parameters
device = 'cuda:0'
n_epochs = 100
runs = 1
patience = 3

lr = 0.001 # learning rate
clip = 5 # Clip the gradient


def main():

    # Preprocess and load data
    train_loader, dev_loader, test_loader, lang, tokenizer = preprocess_and_load_data()

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    num_intents = len(lang.intent2id)
    num_slots = len(lang.slot2id)

    model = BERTMultiTaskModel(bert_model, num_intents, num_slots)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    model.to(device)


    results_test, intent_test = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_intents, criterion_slots, model, device, lang, tokenizer, n_epochs, patience)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy: ', intent_test['accuracy'])
    
    #TODO check for multiple runs

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()
    
