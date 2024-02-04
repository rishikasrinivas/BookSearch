import torch

import numpy as np
from transformers import RobertaConfig
from transformers import AdamW, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RobertaBase():
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 10, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            #output_attentions = False, # Whether the model returns attentions weights.
            #output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", max_length = 512)
        self.config = RobertaConfig()
    def getModel(self):
        return self.model
    
    
    def get_tokenizer(self):
        return self.tokenizer
    
  
       
  
    def tokenize(self, txt):
        return self.tokenizer(txt, return_tensors='pt')
    

       

