from transformers import RobertaConfig
from transformers import AdamW, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from process_data import getDF, get_labels
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from transformers import DataCollatorForTokenClassification
from transformers import get_linear_schedule_with_warmup

# Initializing a RoBERTa configuration
configuration = RobertaConfig()

configuration.num_labels = 10
tokenizer = AutoTokenizer.from_pretrained("roberta-base", max_length = 512)

data_collator = DataCollatorForTokenClassification(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 10, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    #output_attentions = False, # Whether the model returns attentions weights.
    #output_hidden_states = False, # Whether the model returns all hidden-states.
)

def get_input_id_and_attention_masks():
    df = getDF() #from process.py
    
    input_ids = []
    attention_masks = []
    for summ in df['summary']:
        encoded_dict = tokenizer.encode_plus(
                            summ,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 500,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',# Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    

    labels = torch.from_numpy(np.array(df['genre_id'].tolist()))
    return input_ids, attention_masks, labels

input_ids,attention_masks, labels= get_input_id_and_attention_masks()

def createTensorDS(input_ids,attention_masks, labels):
    return TensorDataset(input_ids, attention_masks, labels)

def split(tensorDataset):
    train_size = int(0.85 * len(tensorDataset))
    val_size = len(tensorDataset) - train_size    
    train_dataset, val_dataset = random_split(tensorDataset, [train_size, val_size])
    return train_dataset, val_dataset

def createDataloaders(train_dataset, val_dataset):
    
    batch_size = 16

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    valid_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )
    return train_dataloader, valid_dataloader



def train(model, train, val, epochs):
    total_steps = len(train)*epochs
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)
    total_train_loss = 0
    batch_loss = 0
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        model.train()
        for step, batch in enumerate(train):
            print("train step: ", step)
            input_ids=  batch[0]
            input_mask = batch[1]
            labels = batch[2]
            print(labels)
            out = model(input_ids, attention_mask=input_mask, labels= labels)
            
            loss = out['loss']
            logits =out['logits']
            
            total_train_loss += loss.item()
            batch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
        avg_train_loss = total_train_loss/len(train)
    
   
        print("")
        print("Running Validation...")
        model.eval()
        total_eval_accuracy=0
        total_eval_loss= 0
        num_Eval_steps= 0
        
        for batch in val:
            input_ids= batch[0]
            input_mask=batch[1]
            labels = batch[2]
            
            with torch.no_grad():
                out = model(input_ids,attention_mask=input_mask,labels = labels)
                
                
            loss = out['loss']
            logits = out['logits']
            
            total_eval_loss += loss.item()
            
            logits = logits.detach().numpy()
            label_ids = labels.numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_eval_acc = total_eval_accuracy/len(val) 
        avg_loss_Eval = total_eval_loss/len(val)
        print(
            'epoch: ', epoch,
            'train_loss: ',  avg_train_loss,
            'valid accur ', avg_eval_acc,
            'valid loss ', avg_loss_Eval,
        )

    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)   

input_ids, attention_masks, labels=get_input_id_and_attention_masks() 
ds=createTensorDS(input_ids, attention_masks, labels) 
train_dataset, val_dataset=split(ds)
train_dataloader, valid_dataloader=createDataloaders(train_dataset, val_dataset)
train(model, train_dataloader, valid_dataloader, 3)