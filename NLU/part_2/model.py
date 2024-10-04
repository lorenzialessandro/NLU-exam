import torch
import torch.nn as nn

class JointModel(BertPreTrainedModel):
    '''
    Joint model for intent classification and slot filling
    '''
    def __init__(self, config, intents, slots, dropout_prob=0.1):
        super(JointModel, self).__init__(config)

        self.intents = intents
        self.slots = slots
        self.bert = BertModel(config) # pretrained BERT

        # for param in self.bert.parameters():
        #     param.requires_grad = False # Freeze the BERT parameters

        self.dropout = nn.Dropout(dropout_prob)

        self.intent_out = nn.Linear(self.config.hidden_size, self.intents) # intent classification head
        self.slot_out = nn.Linear(self.config.hidden_size, self.slots) # slot filling head

        # self.apply(self._init_weights)

    def forward(self, token_ids, attention_mask, mapping_slots):
    
        out = self.bert(token_ids, attention_mask=attention_mask) # BERT forward pass

        sequence_output = out[0]  # [batch_size, sequence_length, hidden_size]
        pooled_output = out[1]    # [batch_size, hidden_size]

        # apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # extract the embeddings of the first token corresponding to each word in the sentence
        slot = []
        for i in range(sequence_output.size(0)):
            indices = mapping_slots[i]
            selected = sequence_output[i, indices, :]
            slot.append(selected)
        slot = torch.stack(slot)

        # Compute slot logits for each word
        slot_logits = self.slot_out(slot)  # [batch_size, num_words, slots]
        slot_logits = slot_logits.permute(0, 2, 1)  # to compute the cross-entropy loss => [batch_size, slots, num_words]

        # Compute intent logits using the pooled_output
        intent_logits = self.intent_out(pooled_output)  # [batch_size, intents]

        return slot_logits, intent_logits

