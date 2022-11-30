#!/usr/bin/env python3

import simpleaudio as sa
from rhvoice_wrapper import TTS

tts = TTS(threads=1)

text = """Think of how stupid the average person is, and realize half of them are stupider than that.
Сколько не работай, всегда найдётся козёл, который работает меньше, а получает больше."""

for voice in ["aleksandr+alan", "arina+clb"]:
    data = tts.get(text, format_='wav', voice=voice)
    wave_obj = sa.WaveObject(data, num_channels=1, bytes_per_sample=2, sample_rate=22050)
    play_obj = wave_obj.play()
    play_obj.wait_done()

print("Done! Pess Ctrl+C to exit.")
