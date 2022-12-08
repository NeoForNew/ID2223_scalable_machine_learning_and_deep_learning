import gradio as gr
from transformers import pipeline
from pytube import YouTube

pipe = pipeline(model="NeoonN/ID2223_Lab2_Whisper")

def transcribe(audio, url):
    """
    Transcribes a YouTube video if a url is specified and returns the transcription.
    If not url is specified, it transcribes the audio file as passed by Gradio.
    :param audio: Audio file as passed by Gradio. Only used if no url is specified.
    :param url: YouTube URL to transcribe.
    :param seconds_max: Maximum number of seconds to consider. If the audio file is longer than this, it will be truncated.
    """
    if url:
      video=YouTube(url).streams.filter(only_audio=True).all()
      audio=video[0].download()
      text = pipe(audio)["text"]
      return text

    else:
        text = pipe(audio)["text"]
        return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Transcribe from Microphone"),
        gr.Text(max_lines=1, placeholder="Enter YouTube Link with Chinese speech to be transcribed", label="Transcribe from YouTube URL"),
    ], 
    outputs="text",
    title="Whisper Small Chinese",
    description="Realtime demo for Chinese speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()
