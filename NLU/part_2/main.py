# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

# define parameters
device = 'cuda:0'
hid_size = 200    # size of hidden layer
emb_size = 300    # size of embedding layer
n_epochs = 100
patience = 3

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    lr = 1.9         # learning rate
    # 0.001
    
    
    # Preprocess and load data
    train_loader, dev_loader, test_loader, lang, train_dataset = preprocess_and_load_data()
    vocab_len = len(lang.word2id)
    total_samples = len(train_dataset) 
    
    # Instantiate the model
    emb_size = hid_size # for weight tying 
    model = LM_LSTM_weight_tying(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')   

    # Instantiate the optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = NTAvSGD(model.parameters(), lr=lr, total_samples=total_samples, batch_size=256)
    
    # Train and evaluate the model
    result = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, device, n_epochs, patience)
    
    print(result)
    
    path = 'model_bin/LSTM_VariationalDropout.pt'
    torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
        
