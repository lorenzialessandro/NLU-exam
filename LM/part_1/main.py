# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import sys

# define parameters
device = 'cuda:0'
hid_size = 200    # size of hidden layer
emb_size = 300    # size of embedding layer
n_epochs = 200
runs = 5
patience = 3


lr = 0.0001 # learning rate
clip = 5 # Clip the gradient



def main():
    # Preprocess and load data
    train_loader, dev_loader, test_loader, lang = preprocess_and_load_data()

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)


    # Multiple runs
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                     vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        results_test, intent_test = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, device, n_epochs, patience)

        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
    return round(slot_f1s.mean(),3), round(slot_f1s.std(),3), round(intent_acc.mean(), 3), round(slot_f1s.std(), 3)


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()
