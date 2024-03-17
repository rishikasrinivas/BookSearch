from Process_Input.genre_collect import getTopKGenres
from Model_Dev.model import BERT
from API.api_call import callAPI
from CompareResults.sentence_sim import get_similarity
from transformers import pipeline

def get_books(userInput):
    model = BERT()
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    genres = getTopKGenres(model, userInput, 3)
    print(genres)
    summs = callAPI(genres)
    
    #compare user search to narrowed down results
    
    


    summ=[]
    for val in summs.keys():
        summ.append(val)
    similarities =get_similarity(userInput, summ)
 
    #process inputs with similarity scores
    for (desc, score) in similarities:
        if len(desc.split()) < 10:
            similarities.remove((desc,score))

    titles = {"books_":[]}
    
    for summary in similarities[:5]:
        d1={}

        title= summs[summary[0]]
        d1['title']=title
        d1['summary']=  summarizer(summary[0])[0]['summary_text']
       # d1['categ'] = 
        titles["books_"].append(d1)
        
    print("These books would be of interest to you: ")
    return titles
   #printTitles(titles)

def printTitles(titles):
    i = 1
    for title, summary in titles.items():
        print(i, ": ", title, ": ", summary)
        i+=1


#main("The book is about 11 year old Harry Potter, who receives a letter saying that he is invited to attend Hogwarts, school of witchcraft and wizardry. He then learns that a powerful wizard and his minions are after the sorcerer's stone that will make this evil wizard immortal and undefeatable.")