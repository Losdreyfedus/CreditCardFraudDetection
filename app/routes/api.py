from flask import jsonify, request
from app import app
from app.models.fraud_detection import predict_transaction

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        prediction = predict_transaction(data)
        return jsonify({
            'success': True,
            'is_fraud': prediction['is_fraud'],
            'fraud_probability': prediction['fraud_probability'],
            'risk_score': prediction['risk_score']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400 