import torch
import torch.nn as nn

# intent classification and slot filling

class BERTMultiTaskModel(nn.Module):
    def __init__(self, bert_model, num_intents, num_slots):
        super(BERTMultiTaskModel, self).__init__()
        self.bert = bert_model

        # freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_out = outputs[0]
        pooled_out = outputs[1]  # Use the [CLS] token representation for classification

        sequence_out = self.dropout(sequence_out)
        pooled_out = self.dropout(pooled_out)

        intent_logits = self.intent_classifier(pooled_out)
        slot_logits = self.slot_classifier(sequence_out)

        return intent_logits, slot_logits

