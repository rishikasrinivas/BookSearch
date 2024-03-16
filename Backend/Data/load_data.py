

def loadFileIntoDict(file):
    with open(file, "r") as f:
        lines = f.readlines()
    f.close()
    
    D = {}
    for index_genre_pair in lines:
        index, genre = index_genre_pair.split(',') 
        D[int(index)] = genre[:-1]
    return D
