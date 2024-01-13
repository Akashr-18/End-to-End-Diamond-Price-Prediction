from DiamondRegressor.utils.common import read_yaml, load_object
import os
import joblib
from DiamondRegressor.config.configuration import ConfigurationManager
from DiamondRegressor.components.model_training import Training
from DiamondRegressor.pipeline.stage_03_model_training import ModelTrainingPipeline



def form_response(data):
    print("Data: ", data)
    print("Data type: ", type(data))

    kmeans = load_object(os.path.join('c_obj','preprocessor.pkl'))
    data_in = [[1.52,4,3,5,63,57,12800,7,7,5]]
    cluster_kmeans = kmeans.predict(data_in)
    print("cluster_kmeans: ", cluster_kmeans)


    data_list = data.values().tolist()
    print("Data List: ", data_list)
    

    






    # config = read_yaml(config_path)
    # model = joblib.load(model_dir)
    # prediction = model.predict(data).tolist()[0]
    # return prediction