from flask import Blueprint, render_template, jsonify, url_for, request
import pickle   
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import logging
import shap
import traceback
import requests

site = Blueprint('site', __name__)

MODEL = pickle.load(open('best_model_2.pkl', 'rb'))

try:
    X = pickle.load(open('X.pkl', 'rb'))
except:
    # Create a client object
    client = storage.Client()

    # Retrieve the bucket object
    bucket = client.get_bucket('project-scoring')

    # Retrieve the blob object
    blob = bucket.blob('X.pkl')

    # Download the file to a local file
    blob.download_to_filename('X.pkl')
    X = pickle.load(open('X.pkl', 'rb'))


@site.route('/', methods=['GET', 'POST'])
def index():
    customer_id = request.form.get('customer_id')
    # customer_id="100002"
    if customer_id is not None:
        api_route = url_for('site.api', customer_id=customer_id, _external=True)
        response = requests.get(api_route)
        plot_data = response.json()
    else:
        plot_data = None
    # Render the Plotly JS graph using the plot data
    return render_template('landing.html', plot_data=plot_data)

def reload_X():
    df = pd.read_csv('df.csv')
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(df.median(), inplace=True)
    X = df.drop(columns=["TARGET"])
    X = df.drop(columns=["SK_ID_CURR"])
    # X['id'] = df['SK_ID_CURR']
    scaler = StandardScaler()
    cols = list(X.columns.to_list())
    # cols.remove('id')
    X = pd.DataFrame(scaler.fit_transform(X[cols]), columns=cols)
    # X['SK_ID_CURR'] = df['SK_ID_CURR']
    X['id'] = df['SK_ID_CURR']
    pickle.dump(X, open('X.pkl', 'wb'))
    return

cols = list(X.columns.to_list())
cols.remove('id')
cols.remove('TARGET')

@site.route('/api/<customer_id>')
def api(customer_id):
    # reload_X()
    customer_id_float = int(float(customer_id))
    logging.info(f"customer_id: {customer_id} - Data: {X[X['id']==customer_id_float]}")
    try:
        prediction = MODEL.predict(X.loc[X['id']==customer_id_float, cols])
        pred_ = float(prediction[0])
        probabilities = MODEL.predict_proba(X.loc[X['id']==customer_id_float, cols])
        proba_ = float(probabilities[0][1])
        logging.info(f"customer_id: {customer_id} - Prediction: {pred_} ({type(pred_)}) - Probabilities: {proba_}")
        # Compute SHAP values and extract plot data
        feature_names = np.array(cols)
        explainer = shap.Explainer(lambda x: MODEL.predict_proba(x), X[cols], feature_names=feature_names, seed=42)
        shap_values = explainer(X.loc[X['id'] == customer_id_float, cols], max_evals=3000)
        top_10_shap_dict = {}
        shap_predict = shap_values.base_values[0][1]
        for i in range(len(shap_values.values[0])):
            top_10_shap_dict[cols[i]] = shap_values.values[0][i][1]
            shap_predict += shap_values.values[0][i][1]
        top_10_shap_dict_ordered = {k: v for k, v in sorted(top_10_shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
        top_10_shap_dict_ordered = dict(list(top_10_shap_dict_ordered.items())[:10])
        logging.info(f"customer_id: {customer_id} - Top 10 SHAP Values: {top_10_shap_dict_ordered}")
        logging.info(f"customer_id: {customer_id} - SHAP Predict: {shap_predict}")
        return jsonify({'customer_id': customer_id, 'prediction': pred_, 'probability_default':proba_, 'error':False, 'error_message':'', 'top_10_shap_dict_ordered':top_10_shap_dict_ordered, 'base_shape_proba_default': shap_values.base_values[0][1], 'shap_predict': shap_predict})
    except Exception as e:
        prediction = None
        logging.error(f"customer_id: {customer_id} - Error: {e} - traceback: {traceback.format_exc()}")
        return jsonify({'customer_id': customer_id, 'prediction': 'Could not find customer', 'probability_default':'', 'error':True, 'error_message': f'{e}'})
    
    