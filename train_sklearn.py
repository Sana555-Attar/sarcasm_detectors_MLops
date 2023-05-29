import os
import time
from pathlib import Path
from uuid import uuid4

import joblib
from clearml import Dataset, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
from clearml import Task
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils import plot_confusion_matrix


class SklearnTrainer():
    def __init__(self, model='LinearRegression', seed=42, subset_size=0):
        self.task = Task.init(
            project_name="sarcasm_detector",
            task_name="Sklearn Training",
            output_uri=True
        )
        self.task.set_parameter("model", model)
        self.task.set_parameter("seed", seed)
        self.task.set_parameter("subset_size", subset_size)

        self.seed = seed
        self.model = model
        self.subset_size = subset_size
        self.pipeline = self.create_pipeline()
    
    def create_pipeline(self):
        # Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

        # Model
        if self.model == "LinearRegression":
            # multinomial logistic regression a.k.a softmax classifier
            cfg = {
                "C": 1,
                "n_jobs": 4,
                "solver": 'lbfgs',
                "random_state": 17,
                "verbose": 1
            }
            self.task.connect(cfg)
            model = LogisticRegression(
                **cfg
            )
        else:
            model = None
        
        # Pipeline
        return Pipeline([('vectorizer', vectorizer), ('model', model)])

    def get_data(self):
                          
             local_dataset_path = Path(Dataset.get(dataset_project="sarcasm_detector",
                                                   dataset_name="sarcasm_dataset",
                                                   alias="sarcasm_dataset").get_local_copy())
             data_files=[str(local_dataset_path / csv_path) for csv_path in os.listdir(local_dataset_path)]
             data = []
             for filename in data_files:
                 df = pd.read_csv(filename)#, sep='\t')
                 data.append(df)
             data = pd.concat(data, axis=0, ignore_index=True)

             if self.subset_size:
                   data = data[:self.subset_size]
             ratio = 9/10
             training_sentences = data.iloc[:int(self.subset_size * ratio),0] 
             testing_sentences = data.iloc[int(self.subset_size * ratio):,0]
             training_labels = data.iloc[:int(self.subset_size * ratio),1] 
             testing_labels = data.iloc[int(self.subset_size * ratio):,1]
             return (training_sentences,training_labels,testing_sentences,testing_labels)
    
    
    def train(self):
        train, y_train, test, y_test = self.get_data()

        start_training = time.time()
        self.pipeline.fit(train, y_train)
        self.task.get_logger().report_single_value("train_runtime", time.time() - start_training)

        y_pred = self.pipeline.predict(test)
        self.task.get_logger().report_single_value("Accuracy", accuracy_score(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)

         ##sana added line
        with open("./sklearn_metrics.txt","w") as outfile:
             outfile.write("Accuracy:" +str(accuracy)+"\n")


        self.task.get_logger().report_scalar(title='Performance', series='Accuracy',value=accuracy,iteration=0)

        

        current_dir = os.getcwd()
        folder_name = "sklearn_confusion_matrix"
        file_extension = ".png"

        # Create a timestamp for the file name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        file_name = f"confusion_matrix_{timestamp}{file_extension}"

        path_to_save_folder = os.path.join(current_dir, folder_name)
        os.makedirs(path_to_save_folder, exist_ok=True)

        path_to_save_file = os.path.join(path_to_save_folder, file_name)

        plot_confusion_matrix(
            y_test,
            y_pred,
            ["NORMAL", "SARCASTIC"],
            figsize=(8, 8),
            title=f"{self.model} Confusion Matrix",
            path_to_save_fig=path_to_save_file
            #sana added line above
        )

        os.makedirs("my_awesome_model", exist_ok=True)
        joblib.dump(self.pipeline, f"my_awesome_model/sklearn_classifier_{uuid4()}.joblib")



if __name__ == '__main__':
    sarcasm_trainer = SklearnTrainer(subset_size=1000)
    sarcasm_trainer.train()