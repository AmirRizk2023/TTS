import streamlit as st
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
import torch
import io

# تحميل النماذج
@st.cache_resource
def load_models():
    tts = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
    return tts, vocoder

tts, vocoder = load_models()

# واجهة المستخدم
st.title("🗣️ Text-to-Speech (SpeechBrain)")
text = st.text_area("💬 أدخل نصًا ليتم تحويله إلى صوت:", "This is a test using SpeechBrain and Streamlit.")

# سرعة التشغيل
speed = st.slider("⚡ سرعة التشغيل", 0.5, 2.0, 1.0, 0.1)

if st.button("🎧 استمع"):
    with st.spinner("⏳ جاري التحويل..."):
        # تحويل النص إلى ميل سبيكترجرام
        mel_output, mel_length, alignment = tts.encode_text(text)

        # تحويل الميل إلى صوت
        waveform = vocoder.decode_batch(mel_output)

        # تطبيق تغيير السرعة
        new_sample_rate = int(22050 * speed)

        # حفظ في ذاكرة مؤقتة لتشغيله مباشرة
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform.squeeze(1), new_sample_rate, format="wav")
        buffer.seek(0)

        st.audio(buffer, format="audio/wav")
