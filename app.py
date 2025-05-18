import streamlit as st
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
import torch
import io
import math

# تحميل النماذج
@st.cache_resource
def load_models():
    tts = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
    return tts, vocoder

tts, vocoder = load_models()

# تقسيم النص إلى أجزاء أكبر (150 كلمة)
def split_text(text, max_words=150):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# واجهة المستخدم
st.title("🗣️ Text-to-Speech (SpeechBrain)")
text = st.text_area("💬 أدخل نصًا ليتم تحويله إلى صوت:", "This is a test using SpeechBrain and Streamlit.")

speed = st.slider("⚡ سرعة التشغيل", 0.5, 2.0, 1.0, 0.1)

if st.button("🎧 استمع"):
    with st.spinner("⏳ جاري التحويل..."):
        segments = split_text(text, max_words=150)
        final_waveform = []

        for segment in segments:
            mel_output, mel_length, alignment = tts.encode_text(segment)
            waveform = vocoder.decode_batch(mel_output)
            final_waveform.append(waveform.squeeze(1))

        # دمج جميع الأجزاء الصوتية في موجة واحدة
        combined_waveform = torch.cat(final_waveform, dim=1)

        # تطبيق السرعة المطلوبة
        new_sample_rate = int(22050 * speed)
        buffer = io.BytesIO()
        torchaudio.save(buffer, combined_waveform, new_sample_rate, format="wav")
        buffer.seek(0)

        st.audio(buffer, format="audio/wav")
