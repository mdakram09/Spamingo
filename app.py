from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, url_for, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    #return 'Hello World this is a ML App'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv(
        'Youtube-Spam-Merged.csv')
    data_feature = data[['CONTENT', 'CLASS']]

    data_x = data_feature['CONTENT']
    data_y = data_feature.CLASS

    cv = CountVectorizer()
    X = cv.fit_transform(data_x)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data_y, test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        pred = clf.predict(vect)

    return render_template('index.html',prediction = pred)

if __name__ == '__main__':
    app.run(debug=True)
