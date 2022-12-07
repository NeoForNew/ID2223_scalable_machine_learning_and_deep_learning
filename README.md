# ID2223
## Lab1
## Lab2
Task 1: Fine-tune a model for language transcription, add a UI
● Fine-Tune a pre-trained transformer model and build a serverless UI for using that model
● First Steps
a. Create a free account on huggingface.com
b. Create a free account on google.com for Colab
● Tasks
a. Fine-tune an existing pre-trained transformer model for the Swedish
language, such as Whisper
b. Build and run an inference pipeline with a Gradio UI on Hugging Face
Spaces for your model.
● A sample Colab Notebook is available here.
● Here is a blog post explaining the example code
● You should fine-tune the model with either Swedish or your mother tongue. https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/vi ewer/sv-SE/train
● We recommend that you train your model with a GPU. Colab provides free GPUs for 12 hours (then it shuts down) - so make sure to save your model weights before it shuts down. Getting a good GPU (e.g., p100) is getting harder on Colab. Alternatively, you could modal.com to train your GPU, but i think they will quickly use up your $30 of free credit. Colab is free. If you have your own GPU, you can, of course, use that. You may need to reduce the number of training epochs or dataset size to train in time.
● Communicate the value of your model to stakeholders with an app/service that uses the ML model to make value-added decisions
Example UIs:
● Allow the user to speak into the microphone and transcribe what he/she says (lower grade, as this code is in the example code)
● Allow the user to paste in the URL to a video, and transcribe what is spoken in the video (higher grade)
● Your own creative idea for how to allow people to use your model (highest grade)
1. Describe in your README.md program ways in which you can improve model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to train a better model that one provided in the blog post
If you can show results of improvement, then you get the top grade.
2. Refactor the program into a feature engineering pipeline, training pipeline, and an inference program (Hugging Face Space) to enable you to run feature engineering on CPUs and the training pipeline on GPUs. You should save checkpoints when training, so that you can resume again from when Colab took away your GPU :)
This is challenging - where you can store GBs of data from the feature engineering step? Google Drive? Hopsworks?
