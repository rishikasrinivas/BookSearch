import sys
from flask import Flask, render_template, request
sys.path.append('/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BookSearch/BookSearch/Backend')
from execute import get_books
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/results', methods= ['POST'])
def result():
    query = request.form['query']
    if query != "":
        books = get_books(query)
    else:
        return render_template('index.html', book_list=[])
    
    params = {
        "queries" : query,
        "book_sum": books['books_'],
    }
    if books != []:
        return render_template('index.html', q=params['queries'], b=params['book_sum'])
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)