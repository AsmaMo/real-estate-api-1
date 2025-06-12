from flask import Flask, request, jsonify
from median_voting_regressor import MedianVotingRegressor
import joblib
import numpy as np
import pandas as pd

# من أجل التعرف على MedianVotingRegressor أثناء التحميل
from median_voting_regressor import MedianVotingRegressor

# تحميل البايبلاين المدرب مسبقاً
pipeline = joblib.load("real_estate_pipeline.joblib")

# إنشاء تطبيق Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # قراءة البيانات المرسلة في الطلب
        input_data = request.json

        # تحويل البيانات إلى DataFrame
        input_df = pd.DataFrame([input_data])

        # تنفيذ التنبؤ
        prediction_log = pipeline.predict(input_df)
        prediction = np.expm1(prediction_log)  # نعيد التحويل من log1p

        # إعادة النتيجة
        return jsonify({
            "predicted_price": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# نقطة البداية إذا شغلتها محلياً
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
