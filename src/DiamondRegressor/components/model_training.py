import pandas as pd
import numpy as np
from pathlib import Path
from DiamondRegressor.entity.config_entity import ModelTrainingConfig
from DiamondRegressor.utils.common import create_directories, save_object
from DiamondRegressor import logger
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from kneed import KneeLocator
import matplotlib.pyplot as plt
import os
import joblib
from DiamondRegressor.components.model_finder import ModelFinder
from DiamondRegressor.entity.config_entity import DataPreprocessingConfig, ModelTrainingConfig

class Training:
    def __init__(self , 
                 config:DataPreprocessingConfig,
                 train_config: ModelTrainingConfig):
        self.config = config
        self.train_config = train_config
    
    def elbow_plot(self):
        datapath = self.config.data_file_path
        data = pd.read_csv(datapath)
        logger.info(f"Loaded data of shape: {data.shape} for the elbow plot")
        
        wcss = []
        for i in range(1,11):
            kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1,11)  , wcss)
        plt.title('Elbow Methos')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')

        elbow_plot_file = self.train_config.plot_file_path
        elbow_plot_file = Path(elbow_plot_file)
        plt.savefig(elbow_plot_file)
        logger.info(f"Elbow plot is successfully created and saved at {elbow_plot_file}")

        kn = KneeLocator(range(1,11), wcss, curve='convex', direction='decreasing')
        no_of_clusters = kn.knee
        return no_of_clusters

    def create_cluster_specific_models(self):

        datapath = self.config.data_file_path
        data = pd.read_csv(datapath)
        logger.info(f"Loaded data of shape: {data.shape}")

        no_of_clusters = self.elbow_plot()
        logger.info(f"Optimal number of clusters from elbow plot is {no_of_clusters}")

        logger.info("Clustering started")
        print(data.sample(2))
        kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', n_init=10, random_state=42)
        y_kmeans = kmeans.fit_predict(data)
        save_object(os.path.join('c_obj','preprocessor.pkl'), kmeans)

        logger.info("Clustering completed successfully")
        data['Cluster'] = y_kmeans
        print(data.sample(2))
        clusters = len(data['Cluster'].unique())
        logger.info(f"Unique clusters created in the data: {data['Cluster'].unique()}")
        logger.info(f"Updated data shape with cluster column being added: {data.shape}")

        logger.info("Model Training started for all clusters")
        for i in range(clusters):
            logger.info(f'Entering Cluster {i}')
            cluster_data = data[data['Cluster'] == i]
            cluster_data = cluster_data.drop(['Cluster'], axis=1)
            X = cluster_data.drop('price', axis=1)
            y = cluster_data['price']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            model, model_name = ModelFinder.find_best_model_for_cluster(x_train, x_test, y_train, y_test)
            logger.info(f"Cluster{i}: Best model is {model_name}")
            cluster_no = f'cluster_{i}'
            model_foder = self.train_config.root_dir
            folder_name = os.path.join(model_foder, cluster_no)
            create_directories([folder_name])
            file_name = f"{model_name}.joblib"
            model_path = os.path.join(folder_name, file_name) 
            joblib.dump(model, model_path)
            logger.info(f"Cluster{i}: Model successfully saved at {model_path}")
            logger.info("*******************************************************************")
    

    



        
        
