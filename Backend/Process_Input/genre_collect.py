import sys
sys.path.append('/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/')

from Data.constants import D, MODEL_WEIGHTS

import torch
  

def getTopKGenres(model, txt, k):
    if (txt == ""):
        print("Please enter a valid input")
        return
    model_m = model.getModel()

    token= model.get_tokenizer()
    model_m.load_state_dict(torch.load(MODEL_WEIGHTS,map_location=torch.device('cpu')))

    tokens=token(txt, return_tensors='pt')

    res = model_m(**tokens)
    preds = (-res['logits'].detach().cpu().numpy())[0].argsort()[:k]
    #get the highest pred and all the ones that are75-100% of the highest val
    topk = [D[index] for index in preds]

        
    
    return topk


