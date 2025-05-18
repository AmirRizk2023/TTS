import streamlit as st
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

# تحميل الموديل
@st.cache_resource
def load_models():
    tts = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
    return tts, vocoder

tts, vocoder = load_models()

st.title("🗣️ Text to Speech App")
text = st.text_area("Enter your text below:", "This is a demo using speechbrain and Streamlit.")

if st.button("Generate Speech"):
    with st.spinner("Generating..."):
        mel_output, mel_length, alignment = tts.encode_text(text)
        waveforms = vocoder.decode_batch(mel_output)
        output_file = "output.wav"
        torchaudio.save(output_file, waveforms.squeeze(1), 22050)
        st.audio(output_file)
