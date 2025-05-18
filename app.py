import streamlit as st
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
import torch
import io
import time

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

# دالة لتقسيم النص إلى أجزاء صغيرة
def split_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# تحويل النص إلى صوت
def text_to_speech(text, speed):
    # تقسيم النص إلى أجزاء
    text_parts = split_text(text)
    audio_parts = []
    
    for part in text_parts:
        # تحويل النص إلى ميل سبيكترجرام
        mel_output, mel_length, alignment = tts.encode_text(part)

        # تحويل الميل إلى صوت
        waveform = vocoder.decode_batch(mel_output)

        # تطبيق تغيير السرعة
        new_sample_rate = int(22050 * speed)

        # حفظ الصوت في الذاكرة المؤقتة
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform.squeeze(1), new_sample_rate, format="wav")
        buffer.seek(0)
        audio_parts.append(buffer)
    
    return audio_parts

if st.button("🎧 استمع"):
    with st.spinner("⏳ جاري التحويل..."):
        start_time = time.time()
        
        # تحويل النص إلى صوت
        audio_parts = text_to_speech(text, speed)
        
        # دمج الأجزاء الصوتية
        combined_audio = io.BytesIO()
        for part in audio_parts:
            combined_audio.write(part.read())
        
        combined_audio.seek(0)
        end_time = time.time()
        
        st.audio(combined_audio, format="audio/wav")
        st.write(f"⏱️ تم التحويل في {end_time - start_time:.2f} ثانية")
