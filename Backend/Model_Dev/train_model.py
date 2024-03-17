from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer
import torch
from process_data import getDF
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, SequentialSampler

from transformers import DataCollatorForTokenClassification
from transformers import get_linear_schedule_with_warmup
from sampler import BalanceSampler
NUM_CLASSES = 13
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Modify the output layer to match the number of classes
model.classifier = torch.nn.Linear(in_features = 768, out_features= NUM_CLASSES)
print(model)
data_collator = DataCollatorForTokenClassification(tokenizer)

def get_input_id_and_attention_masks():
    df = getDF() #from process.py
    
    input_ids = []
    attention_masks = []
    for summ in df['summary']:
        encoded_dict = tokenizer.encode_plus(
                            summ,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
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
                sampler = BalanceSampler(train_dataset), 
                batch_size = batch_size 
            )

    valid_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )
    return train_dataloader, valid_dataloader


def calc_accuracy(logits,labels):
    label=[]
    num_ones = 0
    acc = 0
    for label_set in labels:
        labs = []
        for ind, res in enumerate(label_set):
            if res.item() == 1:
                labs.append(ind)
        label.append(labs)
        num_ones += len(labs)

    for i,log in enumerate(logits):
        top_out = (-log).argsort()[:5]
    
        for ind in top_out:
            if ind in label[i]:
                acc = acc+1
    return acc/num_ones

def train(model, train, val, epochs):
    total_steps = len(train)*epochs
    optimizer = torch.optim.Adam(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    for epoch in range(3):
        total_train_loss = 0
        batch_loss = 0
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        model.train()
        for step, batch in enumerate(train):
            input_ids=  batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask=input_mask)

            logits =out['logits']
            loss = loss_fn(logits, labels)
        
            acc += calc_accuracy(logits, labels)
            total_train_loss += loss.item()
            batch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
   
        avg_train_loss = total_train_loss/len(train)
        print('train_loss: ',  avg_train_loss,)
        print('train_acc: ', acc)
        print("Running Validation...")
        model.eval()
        total_eval_accuracy=0
        total_eval_loss= 0
        num_Eval_steps= 0

        for batch in val:
            input_ids= batch[0].to(device)
            input_mask=batch[1].to(device)
            labels = batch[2].to(device)
            with torch.no_grad():
                out = model(input_ids,attention_mask=input_mask)



            logits = out['logits']
            loss = loss_fn(logits, labels)
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            
        avg_loss_Eval = total_eval_loss/len(val)
        print(
            'epoch: ', epoch,
            'train_loss: ',  avg_train_loss,
            'valid loss ', avg_loss_Eval,
        )
input_ids, attention_masks, labels=get_input_id_and_attention_masks() 
ds=createTensorDS(input_ids, attention_masks, labels) 
train_dataset, val_dataset=split(ds)
train_dataloader, valid_dataloader=createDataloaders(train_dataset, val_dataset)
train(model, train_dataloader, valid_dataloader, 3)