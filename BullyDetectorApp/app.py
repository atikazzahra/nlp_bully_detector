from flask import Flask, render_template, request, send_from_directory
from nlp import tfidf_fit, tfidf_transform, preprocess, get_tweets, evaluate
import re

app = Flask(__name__)


@app.route("/")
def main():
	return render_template('index.html')

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@app.route('/results', methods=['POST'])
def results():
    if 'check_tweet' in request.form.keys():
        tweet = request.form['check_tweet']
        instances = [tweet]
        result = evaluate(instances)
    else:
        username = request.form['check_user']
        public_tweets = get_tweets(username)
        result = evaluate(public_tweets)
        result['username'] = username
    return render_template("result.html", result = result)

if __name__ == "__main__":
	app.run()