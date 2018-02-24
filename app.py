import Algorithmia
from flask import Flask
from flask import request
from sklearn.externals import joblib
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello():
    message = request.form["message"]
    print(message)
    input = {
        "document": message
    }
    corpus = [message]
    client = Algorithmia.client('simrdlXrUKTiKeVsEYIaiuQVa7/1')
    algo = client.algo('nlp/SentimentAnalysis/1.0.4')
    sentiment = algo.pipe(input).__getattribute__("result")[0]["sentiment"]
    cv = joblib.load("cv.pkl")
    sc = joblib.load("sc.pkl")
    classifier = joblib.load("classifier.pkl")
    X = cv.transform(corpus).toarray()
    X = sc.transform(X)
    pred = classifier.predict_proba(X)
    input = {
        "document": corpus[0]
    }
    client = Algorithmia.client('simrdlXrUKTiKeVsEYIaiuQVa7/1')
    algo = client.algo('nlp/SentimentAnalysis/1.0.4')
    sentiment = algo.pipe(input).__getattribute__("result")[0]["sentiment"]
    print(str(pred) + " " + str(sentiment))
    if pred[0][1] > .5:
        return str(1)
    if pred[0][1] > .4 and sentiment < -.2:
        return str(1)
    if pred[0][1] > .35 and sentiment < -.45:
        return str(1)
    if sentiment - pred[0][1] < -.65 and pred[0][1] > .15:
        return str(1)
    return str(0)


if __name__ == "__main__":
    app.run()
