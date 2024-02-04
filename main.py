from Backend.Process_Input.genre_collect import getTopKGenres
from Backend.Model_Dev.model import RobertaBase
from Backend.API.api_call import callAPI
from Backend.CompareResults.sentence_sim import get_similarity

def main(userInput):
    model = RobertaBase()
    genres = getTopKGenres(model, userInput, 1)
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

    titles = {}
    for summary in similarities[:5]:
        title= summs[summary[0]]
        titles[title] = summary[0]
        
    print("These books would be of interest to you: ")
    printTitles(titles)

def printTitles(titles):
    i = 1
    for title, summary in titles.items():
        print(i, ": ", title, ": ", summary)
        i+=1


main("2 siblings go on an adventure")