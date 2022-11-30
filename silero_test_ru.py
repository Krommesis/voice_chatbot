#!/usr/bin/env python3

import torch
import simpleaudio as sa
from scipy.io.wavfile import write

language = 'ru'
model_id = 'v3_1_ru'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to('cpu')  # cpu or cuda
#print(model.speakers)

speaker = model.speakers[3] #5
sample_rate = 48000
put_accent=True
put_yo=False

text = "Сколько не работай, всегда найдётся козёл, который работает меньше, а получает больше."

for i in range(5):
    print(i)
    speaker = model.speakers[i]

    audio = model.apply_tts(text=text,
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo)


    wav = (audio * 32767).numpy().astype('int16')
    play_obj = sa.play_buffer(wav, 1, 2, sample_rate)
    play_obj.wait_done()
