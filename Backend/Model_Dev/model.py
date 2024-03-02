import torch

from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorForTokenClassification
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RobertaBase():
    def __init__(self):
        self.num_classes = 14
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# Load pre-trained model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Modify the output layer to match the number of classes
        self.model.classifier = torch.nn.Linear(in_features = 768, out_features= self.num_classes)
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
    def getModel(self):
        return self.model
    
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize(self, txt):
        return self.tokenizer(txt, return_tensors='pt')
    

       

