import Algorithmia
from flask import Flask
from flask import request
from sklearn.externals import joblib
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def hello():
    message = ""
    if request.method == 'GET':
        message = request.args.get('message')
    else:
        message = request.form["message"]
    print(message)
    input = {
        "document": message
    }
    corpus = [message]
    client = Algorithmia.client('simrdlXrUKTiKeVsEYIaiuQVa7/1')
    algo = client.algo('nlp/SentimentAnalysis/1.0.4')
    sentiment = algo.pipe(input).__getattribute__("result")[0]["sentiment"]
    cv = joblib.load("cv1.pkl")
    sc = joblib.load("sc1.pkl")
    classifier = joblib.load("classifier1.pkl")
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
    expected_value = pred[0][1] + pred[0][2] * 2;
    expected_value += sentiment/-2
    print(expected_value)
    expected_value = round(expected_value)
    print(expected_value)
    if expected_value >= 2:
        return "2"
    elif expected_value >= 1:
        return "1"
    return "0"


if __name__ == "__main__":
    app.run()
