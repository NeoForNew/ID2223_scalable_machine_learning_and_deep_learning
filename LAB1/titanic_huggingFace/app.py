import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def tb_titanic(pclass,sex,age,sibsp,parch,embarked,fare_per_customer,cabin):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(embarked)
    input_list.append(fare_per_customer)
    input_list.append(cabin)
    # 'res' is a list of predictions returned as the label.
    #global res
    res = model.predict(np.asarray(input_list).reshape(1, 8))
    return ("This guy will"+(" survive. " if res[0]=="S" else " die. "))
    
demo = gr.Interface(
    fn=tb_titanic,
    title="Titanic Predictive Analytics",
    description="Predict survivals. 0 for dead and 1 for survived. ",
    inputs=[
        gr.inputs.Number(default=1.0, label="pclass, "),
        gr.inputs.Number(default=1.0, label="gender, male:0, female:1"),
        gr.inputs.Number(default=1.0, label="age"),
        gr.inputs.Number(default=1.0, label="sibsp"),
        gr.inputs.Number(default=1.0, label="parch"),
        gr.inputs.Number(default=1.0, label="embarked, C:1, S:2, Q:3,unknown:0"),
        gr.inputs.Number(default=1.0, label="fare_per_customer"),
        gr.inputs.Number(default=1.0, label="cabin,known:1, unknown:0"),
    ],
    outputs=gr.Textbox()
    )
    # outputs=gr.outputs.Textbox(self,type="auto",label="Hi"))
    #("This guy will"+("survive. " if res[0]==1 else "die. ")
demo.launch()
