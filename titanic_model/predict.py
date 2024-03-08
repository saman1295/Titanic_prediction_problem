import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from titanic_model import __version__ as _version
from titanic_model.config.core import config
from titanic_model.pipeline import titanic_pipe
from titanic_model.processing.data_manager import load_pipeline
from titanic_model.processing.data_manager import pre_pipeline_preparation
from titanic_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = titanic_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = titanic_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'PassengerId':[602],'Pclass':[3],'Name':["Slabenoff, Mr. Petco"],'Sex':['male'],'Age':[0.8],
                'SibSp':[0],'Parch':[0],'Ticket':['349214'],'Cabin':[7.8958,],'Embarked':['S'],'Fare':[29]}
    
    make_prediction(input_data=data_in)
