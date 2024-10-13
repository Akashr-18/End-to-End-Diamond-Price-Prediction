from flask import Flask, render_template, request, jsonify
import os
from prediction_service import predict
from DiamondRegressor.pipeline.prediction_pipeline import PredictPipeline, CustomData

webapp_root = 'webapp'
static_dir = os.path.join(webapp_root, 'static')
template_dir = os.path.join(webapp_root, 'templates')

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                data=CustomData(
                    carat=float(request.form.get("carat")),
                    depth=float(request.form.get("depth")),
                    table=float(request.form.get("table")),
                    x=float(request.form.get("x")),
                    y=float(request.form.get("y")),
                    z=float(request.form.get("z")),
                    cut=request.form.get("cut"),
                    color=request.form.get("color"),
                    clarity=request.form.get("clarity")
                )

                final_data=data.get_data_as_dataframe()
                predict_pipeline=PredictPipeline()
                pred=predict_pipeline.predict(final_data)
                result=round(pred[0],2)
                return render_template('index.html', response=result)
               
        except Exception as e:
            print('Error : %s' % e)
            error = {'error': 'SoMeThInG WeNt WrOnG!!'}
            error = {'error': e}
            return render_template('404.html', error=error)
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
