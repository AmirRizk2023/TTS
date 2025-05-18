import streamlit as st
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
import torch
import io
import re

# تحميل النماذج
@st.cache_resource
def load_models():
    tts = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")
    return tts, vocoder

tts, vocoder = load_models()

# تقسيم النص إلى جمل قصيرة (<=200 حرف)
def split_text(text, max_length=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # تقسيم بالنقط
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# واجهة المستخدم
st.title("🗣️ Text-to-Speech (SpeechBrain)")
text = st.text_area("💬 أدخل نصًا ليتم تحويله إلى صوت:", "Kindness is a simple yet powerful act that can transform lives. A smile, a helping hand, or a kind word can make a world of difference. Let's choose kindness daily, for it costs nothing but means everything. Together, we can create a more compassionate world.")

# سرعة التشغيل
speed = st.slider("⚡ سرعة التشغيل", 0.5, 2.0, 1.0, 0.1)

if st.button("🎧 استمع"):
    with st.spinner("⏳ جاري التحويل..."):
        all_waveforms = []

        # تقسيم النص وتحويل كل جزء على حدة
        for chunk in split_text(text):
            mel_output, mel_length, alignment = tts.encode_text(chunk)
            waveform = vocoder.decode_batch(mel_output)
            all_waveforms.append(waveform)

        # دمج كل المقاطع الصوتية في مقطع واحد
        combined_waveform = torch.cat(all_waveforms, dim=-1)

        # تغيير السرعة
        new_sample_rate = int(22050 * speed)

        # حفظ وتشغيل الملف
        buffer = io.BytesIO()
        torchaudio.save(buffer, combined_waveform.squeeze(0), new_sample_rate, format="wav")
        buffer.seek(0)

        st.audio(buffer, format="audio/wav")
