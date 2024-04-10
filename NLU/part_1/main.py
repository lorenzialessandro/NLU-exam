# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import preprocess_and_load_data

device = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    # define parameters
    hid_size = 200    # size of hidden layer
    emb_size = 300    # size of embedding layer
    lr = 0.001          # learning rate
    
    n_epochs = 100
    patience = 3
    
    
    # Preprocess and load data
    train_loader, dev_loader, test_loader, lang = preprocess_and_load_data()
    vocab_len = len(lang.word2id)
    
    # Instantiate the model
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')   

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Train and evaluate the model
    result = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, device, n_epochs, patience)
    
    print(result)
        
