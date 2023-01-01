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
    df_features = pd.read_csv("https://raw.githubusercontent.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/main/Project/df_Features.csv") 
    aurora_no_fg = fs.create_feature_group(
    name="aurora_predict04",
    version=1,
    description="aurora data",
    primary_key = ["Kp_index","clear_day","clear_night","cloudy","fog","partly_cloudy_day","partly_cloudy_night","rain","snow","wind"]
    )
    aurora_no_fg.insert(df_features,write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
