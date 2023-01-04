# Lab1
1. The Titanic Dataset:
a. https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assi
gnments/lab1/titanic.csv
2. Write a feature pipeline that registers the titantic dataset as a Feature Group with Hopsworks. You are free to drop or clean up features with missing values.
3. Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a binary classifier model to predict if a particular passenger survived the Titanic or not. Register the model with Hopsworks.
4. Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
5. Write a synthetic data passenger generator and update your feature pipeline to allow it to add new synthetic passengers.
6. Write a batch inference pipeline to predict if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
# Lab2
Task 1: Fine-tune a model for language transcription, add a UI
Fine-Tune a pre-trained transformer model and build a serverless UI for using that model
### First Steps
a. Create a free account on huggingface.com
b. Create a free account on google.com for Colab
### Tasks
a. Fine-tune an existing pre-trained transformer model for the Swedish
language, such as Whisper
b. Build and run an inference pipeline with a Gradio UI on Hugging Face
Spaces for your model.
### A sample Colab Notebook is available here.
Here is a blog post explaining the example code
You should fine-tune the model with either Swedish or your mother tongue. https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/vi ewer/sv-SE/train
.We recommend that you train your model with a GPU. Colab provides free GPUs for 12 hours (then it shuts down) - so make sure to save your model weights before it shuts down. Getting a good GPU (e.g., p100) is getting harder on Colab. Alternatively, you could modal.com to train your GPU, but i think they will quickly use up your $30 of free credit. Colab is free. If you have your own GPU, you can, of course, use that. You may need to reduce the number of training epochs or dataset size to train in time.
Communicate the value of your model to stakeholders with an app/service that uses the ML model to make value-added decisions
### Example UIs:
Allow the user to speak into the microphone and transcribe what he/she says (lower grade, as this code is in the example code)
Allow the user to paste in the URL to a video, and transcribe what is spoken in the video (higher grade)
Your own creative idea for how to allow people to use your model (highest grade)
1. Describe in your README.md program ways in which you can improve model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to train a better model than the one provided in the blog post
If you can show results of improvement, then you get the top grade.
2. Refactor the program into a feature engineering pipeline, training pipeline, and an inference program (Hugging Face Space) to enable you to run feature engineering on CPUs and the training pipeline on GPUs. You should save checkpoints when training, so that you can resume again from when Colab took away your GPU :)
This is challenging - where you can store GBs of data from the feature engineering step? Google Drive? Hopsworks?
## Code Explanation
### Feature Pipeline

The [Chinese_whisper_feature_pipeline.ipynb](https://github.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/blob/main/Lab2/Chinese_whisper_feature_pipeline.ipynb) is downloaded from Google Colab and only CPU is used when extracting the features. The data is obtained from [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) and `Subset = zh-CN` for simplified Chinese language. Irrelevant columns are removed and a feature extractor and a tokenizer are used for extracting the spectrogram features and preprocessing the labels respectively. 

The preprocessing of the feature pipeline takes around half an hour each time, making feature storage essential in our case. However, the generated features are quite huge (roughly 50GB) in total, we stored the whole generated feature in Google drive.

### Training Pipeline

In the [Chinese_whisper_training_pipeline.ipynb](https://github.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/blob/main/Lab2/Chinese_whisper_training_pipeline.ipynb) downloaded from Colab, we used the GPU provided by Colab Pro+ to training the final model

The evaluation metric is configured as word error rate (WER) and the pretrained "whisper-small" model is loaded. Our model saves checkpoints to Google Drive. We trained for 4000 steps and saves the checkpoints every 1000 steps. After the training completes, which took roughly 8 hours in total, we reached a WER of 19.89%.


## Interactive UI
With the trained model, we uploaded it to [Hugging face Model](https://huggingface.co/NeoonN/ID2223_Lab2_Whisper/tree/main) and created a [Hugging face interactive UI](https://huggingface.co/spaces/NeoonN/Video_whisper). The application design is available in [huggingface-spaces-whisper/app.py](https://huggingface.co/spaces/NeoonN/Video_whisper/blob/main/app.py).
![image](https://github.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/blob/main/Lab2/ui.jpg)
### Function 1
Users can click on the Record from microphone button to start speaking in Chinese and click on stop recording when finished speaking. After clicking on Submit for a while(less than 30s), the spoken words will be shown on the output box to the right.
### Function 2
Users can input a Youtube URL such as [this](https://www.youtube.com/watch?v=EX5hcbzZCow) and the audio of this youtube video will be transcribed into text- 我在北京（I am in Beijing）.

## Performance improvement
### model-centric approach
For model-centric approach, we can use the following methods:
1. Use a larger model with more layers and train longer e.g. using whisper-medium or whisper-large. 
2. Parameter can be selected using random search and grid search. Batch size, learning rate can be optimized and dropout can be applied.
### Data-centric Approaches
For data-centric approaches, we use other public Chinese audio dataset on OpenSLR such as [THCHS-30](https://www.openslr.org/18/), [Free ST Chinese Mandarin Corpus](https://www.openslr.org/38/), [Primewords Chinese Corpus Set 1](https://www.openslr.org/47/), [aidatatang_200zh](https://openslr.org/62/), [MAGICDATA Mandarin Chinese Read Speech Corpus](https://openslr.org/68/).
### Debug hints
`trainer.push_to_hub(**kwargs)`
```
OSError: Tried to clone a repository in a non-empty folder that isn't a git repository. If you really want to do this, do it manually:
git init && git remote add origin && git pull origin main
 or clone repo to a new folder and move your existing files there afterwards.
```
clone the repo on Huggingface and try again.
If it still doesn't work, mannually upload the model on Huggingface
# Project
## Description
The project focus on using an ML model to predict if the aurora will occur in Kiruna based on Kp and the weather in Kiruna. Hopsworks is used to store the feature group in the CSV file and Huggingface is used to build the interactive App. We tried different machine learning models and the decision tree is selected as our final classier which achieves an accuracy of 0.91 and an AUC score of 0.72.
## Interactive UI
The UI is built using HuggingFace and Gradio API. The [app](https://github.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/blob/main/Project/advanced_app/app.py) uses KP and the weather condition as input and will output a picture to show if the aurora will occur or not.
![image](https://github.com/NeoForNew/ID2223_scalable_machine_learning_and_deep_learning/blob/main/Project/pic/aurora_prediction.jpg)
