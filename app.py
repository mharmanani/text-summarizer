from flask import Flask, render_template, request
from summarizer import *
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/summarize', methods=["POST"])
def summarize():
	res = request.form
	return render_template("summarized.html", result=generate_summary(res['userInput']))

if __name__ == '__main__':
    app.run(debug=True)