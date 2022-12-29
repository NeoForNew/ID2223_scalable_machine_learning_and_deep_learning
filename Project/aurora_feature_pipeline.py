import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    df_features_no_onehot = pd.read_csv("https://raw.githubusercontent.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/main/Project/df_Features_no_onehot.csv") 
    aurora_no_fg = fs.get_or_create_feature_group(
    name="aurora_batch_fg",
    version=1,
    description="aurora data",
    primary_key = ["aurora_label","kp_index", "cloudcover", "visibility", "icon", "conditions"],
    )
    aurora_no_fg.insert(df_features_no_onehot,write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()