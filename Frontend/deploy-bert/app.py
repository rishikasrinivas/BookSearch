from flask import Flask, request, redirect, url_for, render_template
import pickle
app = Flask(__name__)

@app.route("/")
def hello_word():
    return "<p>Hi, </p>"

if __name__=="__main__":
    app.run()

