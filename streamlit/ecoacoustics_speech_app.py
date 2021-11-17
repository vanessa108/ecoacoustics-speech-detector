import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
import tempfile
import numpy as np
from math import ceil
import pandas as pd
from sound import sound
import matplotlib.pyplot as plt
import librosa, librosa.display
def load_wav_16k_mono(filename):
    # """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
         file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# """Summarises results for each minute - use directly with model output"""
def speech_second2minute(speech_seconds):
    speech_num_minutes = ceil(len(speech_seconds)/125)
    speech_minute = np.zeros(speech_num_minutes)
    minutes = np.linspace(0, speech_num_minutes*125, speech_num_minutes+1)
    for i in range(0, len(minutes)-1):
        start_t = int(minutes[i])
        end_t = start_t + 124
        this_min = speech_seconds[start_t:end_t]
        speech = evaluate_minute(this_min)
        speech_minute[i] = speech
    return speech_minute
# """Use sliding window to evaluate each 3*0.48 second segment as speech/not speech
#     True if the entire window has positve results [1 1 1]"""
def evaluate_minute(speech):
    window_size = 3
    windows = np.lib.stride_tricks.sliding_window_view(speech, window_size)
    speech = windows[:, 0] * windows[:, 1] * windows[:, 2]
    speech = np.sum(speech)
    if speech > 0:
        return 1
    return 0
st.set_page_config(page_title='Speech Detector')
# load model 
my_classes = ['not speech', 'speech']
map_class_to_id = {'not speech':0, 'speech':1}
hop_length = 0.48
@st.cache
def load_model():
    path = './speech_detection_model/'
    return tf.saved_model.load(path)
model = load_model()


# uploaded_file = st.file_uploader('File uploader', type=['wav'])
# if uploaded_file != None:
#     f = tempfile.NamedTemporaryFile()
#     f.write(uploaded_file.getbuffer())
#     audio = load_wav_16k_mono(f.name)
#     output = model(audio).numpy()
#     st.write(np.argmax(output))
# input: uploaded file object, output: if a minute contains speech
def process_audio(uploaded_file):
    f = tempfile.NamedTemporaryFile()
    f.write(uploaded_file.getbuffer())
    audio = load_wav_16k_mono(f.name)
    output = model(audio).numpy()
    results = speech_second2minute(output)
    return results, f.name


def main():
    st.title('Speech detector for ecoacoustics')
    upload_container = st.container()
    record_container = st.container()
    uploaded_file = upload_container.file_uploader('Upload a WAV file',  type=['wav'])
    #st.selectbox('Choose an output', ['CSV', 'audio files'])
    if record_container.button('Record'):
        with st.spinner('Recording for 2 seconds....'):
            sound.record()
        st.success("Recording completed")
        audio = load_wav_16k_mono("recorded.wav")
        output = model(audio).numpy()
        speech_detected = speech_second2minute(output)
        minutes = np.arange(0, len(speech_detected))
        results = pd.DataFrame()
        results['Minute'] = minutes
        results['Speech Detected'] = speech_detected
        record_container.dataframe(results)
        
    if uploaded_file:
        speech_detected, filename = process_audio(uploaded_file)
        y, sr = librosa.load(uploaded_file)
        D = librosa.stft(y)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
        ax.set(title='Spectrogram of uploaded audio')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        upload_container.pyplot(fig)
        minutes = np.arange(0, len(speech_detected))
        results = pd.DataFrame()
        results['Minute'] = minutes
        results['Speech Detected'] = speech_detected
        upload_container.dataframe(results)

if __name__ == "__main__":
    main()

# to do:
# print message with minutes where speech is detected 
# offer speech cutting options 
# some markdown nice things 