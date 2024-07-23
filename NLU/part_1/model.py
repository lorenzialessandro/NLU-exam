import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ===============  ===============
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent



# =============== ModelIAS : Adding bidirectionality  ===============
  # allows the model to consider both past and future context when making predictions for each element in a sequence

    # 1. Forward Pass:
      # the input sequence is processed from left to right.
      # At each time step, the model updates its hidden state based on the current input and the previous hidden state.
      # This allows the model to capture dependencies and patterns in the data up to that point in time.

    # 2. Backward Pass:
      # the input sequence is processed from right to left.
      # At each time step, the model updates a separate set of hidden states based on the current input and the subsequent hidden state.
      # This allows the model to capture dependencies and patterns in the data from future time steps.

    # 3. Combination:
      # After both the forward and backward passes are completed, the hidden states from both directions are typically combined in some way to form the final representation of each element in the sequence.
      # This combined representation contains information from both past and future context, allowing the model to make more informed predictions.

class ModelIAS_bidirectional(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_bidirectional, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True) # + bidirectional=True
        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # + Double the hid_size for bidirectional LSTM
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # + Double the hid_size for bidirectional LSTM
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Concatenation of the forward and backward hidden states along the feature dimension (dim=1)
        # to obtain the final hidden state for both slots and intent prediction
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1) # + Concatenate the forward and backward hidden states
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent


# =============== ModelIAS : Adding dropout layer  ===============
# dropout after the LSTM layer and before passing the output to the linear layers
  # nn.Dropout(0.5) is added with a dropout probability of 0.5.
  # The dropout layer is applied after the LSTM layer to the packed output.
  # The dropout layer helps regularize the model by randomly dropping out connections during training, reducing overfitting

class ModelIAS_dropout(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_dropout, self).__init__()


        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # + Adding dropout with 50% probability

        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # Double the hid_size for bidirectional LSTM
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # Double the hid_size for bidirectional LSTM

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to the unpacked sequence
        utt_encoded = self.dropout(utt_encoded) # + Apply dropout to the LSTM output
          # applying dropout to utt_encoded ensures that dropout is applied only to the relevant elements of the sequence,
          # avoiding any unwanted effects on the padded elements and ensuring that the model can learn effectively
          # from the sequence data.

        # Concatenate the forward and backward hidden states
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
