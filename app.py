from flask import Flask, render_template, request, jsonify
import os
from prediction_service import predict

webapp_root = 'webapp'

static_dir = os.path.join(webapp_root, 'static')
template_dir = os.path.join(webapp_root, 'templates')

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                data_dict = dict(request.form)
                response = predict.form_response(data_dict)
                return render_template('index.html', response=response)
            # elif request.json:
            #     response = predict.api_response(request.json)
            #     return jsonify(response)                
        except Exception as e:
            print('Error : %s' % e)
            error = {'error': 'SoMeThInG WeNt WrOnG!!'}
            error = {'error': e}
            return render_template('404.html', error=error)
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)