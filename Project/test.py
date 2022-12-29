import hopsworks

project = hopsworks.login()

fs = project.get_feature_store()