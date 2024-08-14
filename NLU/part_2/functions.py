# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from conll import evaluate
from sklearn.metrics import classification_report

clip = 5

# =============== Model Training and Evaluation Functions ===============

# Performs the training loop.
def training_loop(model, train_loader, optimizer, criterion_intents, criterion_slots, clip, device='cuda:0'):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()

        # Extract input and target data
        input_ids = batch['utterances'].to(device)
        attention_mask = torch.stack(batch['attention_mask']).to(device) # check
        intent_targets = batch['intents'].to(device)
        slot_targets = batch['y_slots'].to(device)

        # Forward pass
        intent_logits, slot_logits = model(input_ids, attention_mask)

        # Compute loss
        intent_loss = criterion_intents(intent_logits, intent_targets) # loss for intents
        slot_loss = criterion_slots(slot_logits.view(-1, slot_logits.shape[-1]), slot_targets.view(-1)) # loss for slots


        loss = intent_loss + slot_loss # total loss

        # print(f"Intent Loss: {intent_loss.item()}, Slot Loss: {slot_loss.item()}")

        total_loss += loss.item()

        # print(f"Total Loss: {total_loss}")

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # clipping
        optimizer.step()

    # average training loss
    return total_loss / len(train_loader)


# handle tokenizer  quotes (')
# used in validation loop
def process_token(token):
    res_token = []

    for word in token:
        if "'" in word:
            tmp = ""
            for w in word:
                if w != "'":
                    tmp += w
                else:
                    if tmp:
                        res_token.append(tmp)
                        tmp = ""
                    res_token.append('O')
            if tmp:
                res_token.append(tmp)
        else:
            res_token.append(word)

    return res_token

# Performs the evaluation loop.
def evaluate_loop(model, dev_loader, criterion_intents, criterion_slots, lang, tokenizer, device='cuda:0'):
    model.eval()

    total_loss = 0.0
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():

        for batch in dev_loader:
            input_ids = batch['utterances'].to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device) # check
            intent_labels = batch['intents'].to(device)
            slots = batch['y_slots'].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)

            # Compute loss for intents
            intent_loss = criterion_intents(intent_logits, intent_labels)

            # Compute loss for slots
            slot_loss = criterion_slots(slot_logits.view(-1, model.num_slots), slots.view(-1))


            # Total loss
            loss = intent_loss + slot_loss

            total_loss += loss.item()

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in intent_labels.tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slot_logits, dim=2)

            #for i in range(output_slots.size(0)):
                #for j in range(output_slots.size(1)):
                    #ref_slots.append(lang.id2slot[slot_labels[i][j].item()])
                    #hyp_slots.append(lang.id2slot[output_slots[i][j].item()])

            for idx, seq in enumerate(output_slots): # check all this cycle

                slots_len = batch['slots_len'].tolist()[idx]

                token = tokenizer.decode(input_ids[idx]) # decode the token

                slots_ids = slots[idx].tolist() # get ids
                slots_labels = [lang.id2slot[elem] for elem in slots_ids[:slots_len]] # get labels
                slots_labels = slots_labels[1:] # [CLS]

                next_slots = seq[1:slots_len].tolist()
                token_split = token.split()

                # Initialize an empty list to store the corrected tokens
                decoded_token = []

                # Iterate over the split tokens
                for token in token_split:
                    # Check if the token is just a single quote (or any other unwanted extra token)
                    if token == "'":
                        # If the last token in the list ends with an alphabetic character, merge them
                        if decoded_token and decoded_token[-1].isalpha():
                            decoded_token[-1] += token
                        else:
                            # Otherwise, add 'O' to indicate an unwanted token
                            decoded_token.append('O')
                    else:
                        # If the token is valid, just add it to the corrected list
                        decoded_token.append(token)

                # Now decoded_token contains the corrected tokens



                # check
                while len(decoded_token) < len(slots_ids):
                    decoded_token.append(lang.slot2id['pad'])

                decoded_token = decoded_token[1:] # [CLS]

                ref_slots.append([(decoded_token[i], j) for i, j in enumerate(slots_labels)])
                hyp_slots.append([(decoded_token[i], lang.id2slot[j]) for i, j in enumerate(next_slots)])



    try:
        # Evaluate slot filling results
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Handle exceptions
        print("Warning:", ex)
        # Print the set difference between reference and predicted slots
        ref_set = set([x[1] for x in ref_slots])
        hyp_set = set([x[1] for x in hyp_slots])
        print("Difference in reference and predicted slots:", hyp_set.difference(ref_set))
        results = {"total": {"f": 0}}

    # Generate classification report for intents
    report_intent = classification_report(ref_intents, hyp_intents,
                                           zero_division=False, output_dict=True)

    # Calculate average validation loss
    avg_dev_loss = total_loss / len(dev_loader)

    return results, report_intent, avg_dev_loss

# Initializes the weights of the model. It is called during model initialization to set the initial weights
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


# =============== Training and Evaluation Loops ===============
def train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_intents, criterion_slots, model, device, lang, tokenizer, n_epochs=200, patience=3, ):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    
    pbar = tqdm(range(1,n_epochs))


    for epoch in pbar:
        # print(f'Epoch {epoch}')
        loss = training_loop(model, train_loader, optimizer,
                        criterion_intents, criterion_slots, clip=clip)

        if epoch % 1 == 0: 
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = evaluate_loop(model, dev_loader, criterion_intents, criterion_slots, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            pbar.set_description("f1: %f" %f1)
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

        # print('Slot F1: ', results_dev['total']['f'])

    results_test, intent_test, _ = evaluate_loop(model, test_loader, criterion_intents, criterion_slots, lang, tokenizer)
    return results_test, intent_test