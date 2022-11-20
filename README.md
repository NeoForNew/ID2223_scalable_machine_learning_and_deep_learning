# ID2223_scalable_machine_learning_and_deep_learning
## Lab 1
 1. The Titanic Dataset:
a. https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assi
gnments/lab1/titanic.csv
2. Write a feature pipeline that registers the titantic dataset as a Feature Group with Hopsworks. You are free to drop or clean up features with missing values.
3. Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a binary classifier model to predict if a particular passenger survived the Titanic or not. Register the model with Hopsworks.
4. Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
5. Write a synthetic data passenger generator and update your feature pipeline to allow it to add new synthetic passengers.
6. Write a batch inference pipeline to predict if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger pre
### Install dependencies
`pip install -r requirements.txt`
