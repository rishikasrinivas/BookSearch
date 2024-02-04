import requests

def callAPI(genres):
#call model yo predict genre and get predicted genre
    api_key="AIzaSyB6g-mMjEJc8Wd8Al_p6Z94rnoYvUeAVkk"

    url="https://www.googleapis.com/books/v1/volumes"

    search = f"subject:{'+'.join(genres)}"
    
    api_url=f"{url}?q={search}&maxResults=20&key={api_key}"  
    print(api_url)
    response = requests.get(api_url)
    if response.ok:
        responses = response.json()

        book_summs = []
        descriptions= {}
        book_summs = responses.get("items", []) 
        for book in book_summs:
            volume_info = book.get('volumeInfo', [])
            title = volume_info.get('title', 'Unknown Title')
            description = volume_info.get('description', 'No description available')

            descriptions[description] = title
    else:
        print("Invalid API call")
        return
    return descriptions
    #call similarityComp() by llm on wuery and book_sums

    #get top 5 scores