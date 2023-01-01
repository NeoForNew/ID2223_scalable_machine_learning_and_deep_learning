import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("aurora_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/aurora_model.pkl")

def tb_aurora(Kp_index,cloudcover,visibility,clear_day,
               clear_night,cloudy,fog,partly_cloudy_day,
               partly_cloudy_night,rain1,snow1,wind,Clear,
               Overcast,Partially_cloudy,Rain,Rain_Overcast,
               Rain_Partially_cloudy,Snow,Snow_Overcast,Snow_Partially_cloudy,
               Snow_Rain,Snow_Rain_Overcast):
    input_list = []
    input_list.append(Kp_index)
    input_list.append(cloudcover)
    input_list.append(visibility)
    input_list.append(clear_day)
    input_list.append(clear_night)
    input_list.append(cloudy)
    input_list.append(fog)
    input_list.append(partly_cloudy_day)
    input_list.append(partly_cloudy_night)
    input_list.append(rain1)
    input_list.append(snow1)
    input_list.append(wind)
    input_list.append(Snow)
    input_list.append(Clear)
    input_list.append(Overcast)
    input_list.append(Partially_cloudy)
    input_list.append(Rain)
    input_list.append(Rain_Overcast)
    input_list.append(Rain_Partially_cloudy)
    input_list.append(Snow_Overcast)
    input_list.append(Snow_Partially_cloudy)
    input_list.append(Snow_Rain)
    input_list.append(Snow_Rain_Overcast)
    
    # 'res' is a list of predictions returned as the label.
    #global res
    res = model.predict(np.asarray(input_list).reshape(1, 23))
    return ("This aurora will"+(" occur " if res[0]==0 else " not occur"))
    
demo = gr.Interface(
    fn=tb_aurora,
    title="aurora Predictive Analytics",
    description="Predict aurora 0 for not occur and 1 for occur. ",
    inputs=[
        gr.inputs.Number(default=0.0, label="Kp_index"),
        gr.inputs.Number(default=0.0, label="cloudcover"),
        gr.inputs.Number(default=0.0, label="visibility"),
        gr.inputs.Dropdown(["snow","cloudy","partly-cloudy-night","clear-night","clear-day"]),
        gr.inputs.Dropdown(["Overcast","Partially cloudy","Clear","Snow, Overcast"])
    ],
    outputs=gr.Textbox()
    )
    # outputs=gr.outputs.Textbox(self,type="auto",label="Hi"))
    #("This guy will"+("survive. " if res[0]==1 else "die. ")
demo.launch()
