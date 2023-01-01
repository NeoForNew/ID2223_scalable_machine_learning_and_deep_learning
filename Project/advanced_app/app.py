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
model = mr.get_model("aurora_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/aurora_model.pkl")


def tb_aurora(Kp_index, cloudcover, visibility, icon, conditions):
    input_list = []
    input_list.append(Kp_index)
    input_list.append(cloudcover)
    input_list.append(visibility)
    input_icon = icon
    input_conditions = conditions
    icon_feature_list = ['clear_day', 'clear_night', 'cloudy', 'fog', 'partly_cloudy_day', 'partly_cloudy_night', 'rain',
                 'snow', 'wind']
    conditions_feature_list = ['Clear', 'Overcast', 'Partially_cloudy', 'Rain', 'Rain_Overcast', 'Rain_Partially_cloudy',
                       'Snow', 'Snow_Overcast', 'Snow_Partially_cloudy', 'Snow_Rain', 'Snow_Rain_Overcast']
    icon_feature_list.append(input_icon)
    conditions_feature_list.append(input_conditions)
    icon_df = DataFrame(icon_feature_list)
    icon_df_one = pd.get_dummies(icon_df)
    icon = icon_df_one.values.tolist()[9]

    conditions_df = DataFrame(conditions_feature_list)
    conditions_df_one = pd.get_dummies(conditions_df)
    conditions = conditions_df_one.values.tolist()[11]
    input_list.extend(icon)
    input_list.extend(conditions)
    print(input_list)

    # 'res' is a list of predictions returned as the label.
    # global res
    res = model.predict(np.asarray(input_list).reshape(1, 23))
    return ("This aurora will" + (" occur " if res[0] == 0 else " not occur"))

demo = gr.Interface(
    fn=tb_aurora,
    title="aurora Predictive Analytics",
    description="Predict aurora 0 for not occur and 1 for occur. ",
    inputs=[
        gr.inputs.Number(default=0.0, label="Kp_index"),
        gr.inputs.Number(default=0.0, label="cloudcover"),
        gr.inputs.Number(default=0.0, label="visibility"),
        gr.inputs.Dropdown(['clear_day', 'clear_night', 'cloudy', 'fog', 'partly_cloudy_day', 'partly_cloudy_night', 'rain',
                 'snow', 'wind'], label="icon"),
        gr.inputs.Dropdown(['Clear', 'Overcast', 'Partially_cloudy', 'Rain', 'Rain_Overcast', 'Rain_Partially_cloudy',
                       'Snow', 'Snow_Overcast', 'Snow_Partially_cloudy', 'Snow_Rain', 'Snow_Rain_Overcast'], label="conditions")
    ],
    outputs=gr.Textbox()
    )
demo.launch()