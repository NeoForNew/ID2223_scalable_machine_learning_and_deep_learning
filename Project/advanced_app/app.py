import gradio as gr
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
#model = mr.get_model("aurora_modal", version=1)
#model_dir = model.download()
model = joblib.load("/home/neo/ID2223/Project/aurora_model/aurora_model.pkl")


def tb_aurora(Kp_index,  visibility, icon):
    input_list = []
    input_list.append(Kp_index)
    input_list.append(visibility)
    input_icon = icon

    icon_feature_list = ['clear_day', 'clear_night', 'cloudy', 'fog', 'partly_cloudy_day', 'partly_cloudy_night', 'rain',
                 'snow', 'wind']
    
    icon_feature_list.append(input_icon)

    icon_df = DataFrame(icon_feature_list)
    icon_df_one = pd.get_dummies(icon_df)
    icon = icon_df_one.values.tolist()[9]

    input_list.extend(icon)
    print(input_list)

    # 'res' is a list of predictions returned as the label.
    # global res
    res = model.predict(np.asarray(input_list).reshape(1, 11))
    return ("This aurora will" + (" occur " if res[0] == 0 else " not occur"))

demo = gr.Interface(
    fn=tb_aurora,
    title="aurora Predictive Analytics",
    description="Predict aurora 0 for not occur and 1 for occur. ",
    inputs=[
        gr.inputs.Number(default=0.0, label="Kp_index"),
        gr.inputs.Number(default=0.0, label="visibility"),
        gr.inputs.Dropdown(['clear_day', 'clear_night', 'cloudy', 'fog', 'partly_cloudy_day', 'partly_cloudy_night', 'rain',
                 'snow', 'wind'], label="icon"),
    ],
    outputs=gr.Textbox()
    )
demo.launch()