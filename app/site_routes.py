from flask import Blueprint, render_template, jsonify
import pickle   
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

site = Blueprint('site', __name__)

@site.route('/')
def index():
    return render_template('landing.html')

@site.route('/api/<customer_id>')
def api(customer_id):
    model = pickle.load(open('best_model.pkl', 'rb'))
    df = pd.read_csv('df.csv')
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(df.median(), inplace=True)
    X = df.drop(columns=["TARGET"])
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    pickle.dump(X, open('X.pkl', 'wb'))
    prediction = model.predict(X[X['SK_ID_CURR']==customer_id])
    return jsonify({'customer_id': customer_id, 'prediction': prediction[0]})