import sys
sys.path.append('/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/model_dev')
from Backend.Model_Dev.model import RobertaBase


import torch
fiction = ['fantasy', 'horror', 'thriller', "romance"]
nonfiction = ['science', 'crime', 'history', 'sports', 'travel']
id2genre={0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller', 6: 'psychology', 7: 'romance', 8: 'sports', 9: 'travel'}
def getTopKGenres(model, txt, k):
    if (txt == ""):
        print("Please enter a valid input")
        return
    model_m = model.getModel()

    token= model.get_tokenizer()
    path = '/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/model_weights.pth'
    model_m.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    tokens=token(txt, return_tensors='pt')

    res = model_m(**tokens)
    indices = [i for i in range(10)]
    preds = res['logits'].detach().cpu().numpy()[0]
    #get the highest pred and all the ones that are75-100% of the highest val
    topk = [index for (val, index) in sorted(zip(preds, indices), reverse=True)[:k]]
    genres = [id2genre[i] for i in topk]
    
    print(genres)
    return genres


