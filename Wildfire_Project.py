#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
import eazyml as ez


# In[2]:


username = 'Rajvardhan Ravat'
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhYjgyOWQyYi00MGZhLTQ0ZmEtYTVjMy1lODcxOGY2ZThiMmYiLCJleHAiOjE2MDUyNTE5MzgsImZyZXNoIjpmYWxzZSwiaWF0IjoxNjA1MTY1NTM4LCJ0eXBlIjoiYWNjZXNzIiwibmJmIjoxNjA1MTY1NTM4LCJpZGVudGl0eSI6IlJhanZhcmRoYW4gUmF2YXQifQ.936LvLzRPLzekkc14x4N6w4lMMUGZnZmDODJJESgaqQ'
file_path = 'California_Wildfire_Data.csv'

token = ez.ez_auth(username, None, api_key)["token"]


# In[3]:


def train_data(token, file_path):   
    options = {

        "id": "null",

        "impute": "yes",

        "outlier": "yes",

        "discard": ["started", "extinguished", "counties", "latitude", "longitude", "acresBurned", "visibility", "uvIndex"],

        "accelerate": "yes",
        
        "outcome" : "wildfireIntensity"

    }
    
    response = ez.ez_load(token, file_path, options)
    if response and response["status_code"] != 200:
        print ("Dataset could not uploaded", response["message"])
        return None, None
    dataset_id = response["dataset_id"]
    
    #Now building the model
    options = {

        "model_type" : "predictive",

        "derive_text" : "no",

        "derive_numeric": "no",

        "accelerate": "yes"

    }
    
    response = ez.ez_init_model(token, dataset_id, options)
    if response and response["status_code"] != 200:
        print ("Model could not build", response["message"])
        return None, None
    model_id = response["model_id"]
    best_model = response["model_performance"]["data"][0][0]   
    return model_id, best_model

model_id, best_model = train_data(token, file_path)
print (model_id, best_model)


# In[7]:


predict_file = 'California_Wildfire_Test_Data.csv'
def predict(token, model_id, model_name, predict_file):
    options = {

        "model_name": model_name

    }
    
    response = ez.ez_predict(token, model_id, predict_file, options)
    if response and response["status_code"] != 200:
        print ("Could not predict", response["message"])
        return None
    prediction_id = response["prediction_dataset_id"]
    #options =  {
#
     #   "record_number": list(range(1, len(response['predictions']['indices'])+1))
#
#    }
    #response_explain = ez.ez_explain(token, model_id, prediction_id, options)
    return response
    #prediction_df = pd.DataFrame(response['predictions']['data'], index=response['predictions']['indices'], columns=response['predictions']['columns'])

prediction = predict(token, model_id, best_model, predict_file)
#print (explain)
#print (" ")
print (prediction)


# In[8]:


prediction_id = prediction["prediction_dataset_id"]
options =  {

        "record_number": list(range(1, len(prediction['predictions']['indices'])+1))

    }
response_explain = ez.ez_explain(token, model_id, prediction_id, options)
print (response_explain)


# In[33]:


import pandas as pd
prediction_df = pd.DataFrame(columns= ['started', 'extinguished', 'counties', 'latitude', 'longitude', 'acresBurned', 
                                       'precipIntensity', 'precipIntensityMax', 'precipProbability', 'dewPoint', 'humidity', 
                                       'windSpeed', 'windGust', 'windBearing', 'uvIndex', 'temperatureMin', 'temperatureMax', 
                                       'cloudCover', 'visibility', 'pressure', 'wildfireIntensity', 'Predicted wildfireIntensity',
                                       'explanation'])

for data, rule in zip (prediction['predictions']['data'], response_explain['explanations']):
    
    prediction_df = prediction_df.append({'started': data[0], 'extinguished': data[1], 'counties': data[2], 'latitude': data[3], 
                                          'longitude': data[4], 'acresBurned': data[5], 'precipIntensity': data[6], 
                                          'precipIntensityMax': data[7], 'precipProbability': data[8], 'dewPoint': data[9], 
                                          'humidity': data[10], 'windSpeed': data[11], 'windGust': data[12], 
                                          'windBearing': data[13], 'uvIndex': data[14], 'temperatureMin': data[15], 
                                          'temperatureMax': data[16], 'cloudCover': data[17], 'visibility': data[18], 
                                          'pressure': data[19], 'wildfireIntensity': data[20], 'Predicted wildfireIntensity': data[21],
                                           'explanation' : rule['explanation']},  ignore_index = True)


# In[36]:


prediction_df.to_csv("California_Wildfire_Explanation.csv", index = False)

