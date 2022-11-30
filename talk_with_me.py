#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import word_tokenize
from rhvoice_wrapper import TTS
from num2words import num2words
from time import time

import simpleaudio as sa
import sounddevice as sd
import argparse as ap
import numpy as np
import webrtcvad
import whisper
import queue
import torch
import nltk
import sys
import os
import re


class TalkWithMe(object):
    """docstring for TalkWithMe"""

    def __init__(self, args):
        super(TalkWithMe, self).__init__()
        self.sentencer = None
        self.gpt_model = None
        self.gpt_tokenizer = None
        self.stt_model = None
        self.vad = None
        self.slen = None
        self._pref_ = None
        self.speaker = None
        self.silero_model = None
        self.stt = args.stt
        self.tts = args.tts
        self.brain = args.gpt
        self.gender = args.sex
        self.lang = "en" if args.lang == "en" else "ru"
        self._user_ = args.user
        self._bot_ = args.bot

    @staticmethod
    def load_tokenizer_and_model(model_name_or_path):
        """ (Down)loading selected ruFPT-3 model. """
        return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(
            model_name_or_path).cuda()

    def load_silero_model(self):
        """ (Down)loading silero model if it's defined as TTS. """
        device = torch.device('cpu')  # "cpu" or "cuda"
        model_id = "v3_1_ru" if self.lang == "ru" else "v3_en"

        self.silero_model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self.lang,
            speaker=model_id)
        self.silero_model.to(device)
        if self.lang == "ru":
            # You can define another voice if you want
            # 0 and 4 are male voices, 1, 2 and 3 - female
            # Use silero_test_ru.py to hear them all
            if self.gender == "male":
                self.speaker = self.silero_model.speakers[0]
            else:
                self.speaker = self.silero_model.speakers[1]
        else:
            # There're 0 - 117 voices available for English
            # Use silero_test_en.py to hear them all
            if self.gender == "male":
                self.speaker = self.silero_model.speakers[1]
            else:
                self.speaker = self.silero_model.speakers[0]

    @staticmethod
    def generate(gpt_model, gpt_tok, text,
                 do_sample=True, max_length=128, repetition_penalty=5.0, top_k=5,
                 top_p=0.95, temperature=1., num_beams=None, no_repeat_ngram_size=3):
        """ Generates response based on 2 last strings
                of the dialog and 1 just recognized. """
        input_ids = gpt_tok.encode(text, return_tensors="pt").cuda()
        out = gpt_model.generate(
            input_ids.cuda(), max_length=max_length,
            repetition_penalty=repetition_penalty, do_sample=do_sample,
            top_k=top_k, top_p=top_p, temperature=temperature,
            num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
        return list(map(gpt_tok.decode, out))

    def say_it(self, text: str):
        """ Makes sound of the given text """
        if self.tts == "rhvoice":
            voice = "aleksandr+alan" if self.gender == "male" else "anna+clb"
            # os.system(f'echo "{text}" | RHVoice-test -p {voice}')
            data = TTS(threads=1).get(text, format_='wav', voice=voice)
            wave_obj = sa.WaveObject(data, 1, 2, 22050)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        else:
            audio = self.silero_model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=24000)
            wav = (audio * 32767).numpy().astype('int16')
            play_obj = sa.play_buffer(wav, 1, 2, 24000)
            play_obj.wait_done()

    def listen_to_me(self):
        """ Waits until user says something and
            returns strng of recognized text. """

        def callback(indata, frames, time, status):
            q.put(bytes(indata))

        q = queue.Queue()
        frames = []
        sr = 16000
        with sd.RawInputStream(samplerate=sr, blocksize=480, dtype='int16',
                               channels=1, callback=callback):
            while True:
                frame = q.get()
                voiced = self.vad.is_speech(frame, sr)
                if voiced:
                    frames.append(frame)
                if not voiced and len(frames) > 10:
                    data = b''.join(frames)
                    frames = []
                    data = np.frombuffer(data, dtype=np.int16)
                    data = data.astype("float32") * 0.5 ** 15
                    volume_norm = np.linalg.norm(data)
                    if volume_norm > 6.:
                        audio = whisper.pad_or_trim(data)
                        mel = whisper.log_mel_spectrogram(audio).to(self.stt_model.device)
                        options = whisper.DecodingOptions(language=self.lang)
                        recognized = whisper.decode(self.stt_model, mel, options).text
                        # print(recognized, volume_norm)
                        return recognized

    def load_modules(self):
        """ (Down)loading and setting up needed modules """
        # Voice activity detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        # Speech to text
        print(f"Loading Whispser {self.stt} msodel...")
        start = time()
        self.stt_model = whisper.load_model(self.stt)
        print(f"Whisper model loaded successfully in {time() - start}")
        # Text generator
        print(f"Loading ruGPT-3 {self.brain} model...")
        start = time()
        gpt_model_name = f"sberbank-ai/rugpt3{self.brain}_based_on_gpt2"
        self.gpt_tokenizer, self.gpt_model = self.load_tokenizer_and_model(gpt_model_name)
        print(f"ruGPT-3 model loaded successfully in {time() - start}")
        # Setting up gpt-3 template language, bot and user names.
        if self.lang == 'ru':
            if not self._bot_:
                self._bot_ = "Антон" if self.gender == "male" else "Марина"
            self._user_ = "Хозяин" if not self._user_ else self._user_
            self._pref_ = f"{self._user_}::Вы какого родились?\n{self._bot_}::Числа или хрена?"
            self.sentencer = nltk.data.load('tokenizers/punkt/russian.pickle')
        else:
            if not self._bot_:
                self._bot_ = "Martin" if self.gender == "male" else "Grace"
            self._user_ = "The Lord" if not self._user_ else self._user_
            self._pref_ = f"{self._user_}::Why are you here?\ns{self._bot_}::Well... Heaven didn’t want me, And hells afraid I’ll take over!"
            self.sentencer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.slen = len(self._pref_.split('\n')) + 2
        # Text to speech
        if self.tts == "silero":
            self.load_silero_model()
        else:
            print("Using RHVoice TTS.")

    def main(self):
        self.load_modules()
        print("Ready to go!!! Say something...")
        maxlen = len(word_tokenize(self._pref_)) + 60
        while True:
            # Waiting for recognized text.
            recognized = self.listen_to_me()
            # Now put some commands here.
            # Or create and parse json file with commands you wanna execute.
            # Functions for those commands can be created and imported as modules.
            if recognized.lower() in ["заткнись", "shut up"]:
                print("Exiing...")
                quit()
            # Creating template and generating response.
            text = f"{self._pref_}\n{self._user_}::{recognized}\n{self._bot_}::"
            # Generating text based on created template.
            generated = self.generate(self.gpt_model, self.gpt_tokenizer, text, max_length=maxlen)
            # Clearing the screen after torch logs.
            sys.stdout.write('\033c')
            # Cutting off excess in output.
            # print(generated[0])
            res = generated[0].split('<')[0]
            res = '\n'.join(res.split('\n')[:self.slen])
            res = re.sub("\(.*?\)", "", res,)
            for x in ['.....', '....', '...', '..']:
                res = res.replace(x, ". ")
            # Cutting the last line we got as response.
            response = res.split('\n')[self.slen - 1].split('::')[1]
            # Removing uncomplete sentence if it occures at the end.
            sents = self.sentencer.tokenize(response)
            if len(sents) > 1:
                if sents[-1][-1].isalpha() or sents[-1][-1] in [' ', ',', ';', ':', '-']:
                    res = res.replace(sents[-1], "")
                    response = ' '.join(sents[:-1])
            # Converting numbers into words for silero (it can't read them).
            if any(map(str.isdigit, response)):
                tokens = word_tokenize(response)
                tokens = [t for t in tokens if t.isdigit()]
                for num in tokens:
                    response = response.replace(num, num2words(num, lang=self.lang), 1)
            # Creating new template for next generation.
            self._pref_ = '\n'.join(res.split('\n')[2:self.slen])
            # Recalculating max length of the output.
            maxlen = len(word_tokenize(self._pref_)) + 60  # maximum 60 tokens may be generated in addition to template
            # ": " by default may cause too many spaces after ":", "::" to avoid that.
            print(f"...\n{res.replace('::', ': ')}")
            # Remember, unfortunately, sometimes generator returns empty line.
            # That's why generator takes just 3 lines of the dialog as template.
            # (Anyway, you can modify "self.slen" to increase/decrease this parameter.)
            # The more lines of template the bigger risk to get empty reply.
            # But still it helps bot to keep dialog more or less meanigful
            # at least for a few next phrases without loosing context.
            if response:
                # Let's hear the response. :)
                self.say_it(response)


if __name__ == '__main__':
    # Initializing bot options.
    parser = ap.ArgumentParser(description="A simple voice chatbot program  running from terminal")
    parser.add_argument("-r", "--stt", type=str, default="tiny",
                        help="speech (r)ecognition: tiny/base/small/medium/large")
    parser.add_argument("-b", "--gpt", type=str, default="small",
                        help="(b)rain, ruGPT-3 model: small/medium/large")
    parser.add_argument("-s", "--tts", type=str, default="rhvoice",
                        help="speech (s)ynthesis: rhvoice/silero")
    parser.add_argument("-g", "--sex", type=str, default="female",
                        help="(g)ender of the bot: male/female")
    parser.add_argument("-l", "--lang", type=str, default="ru",
                        help="(l)anguage of the bot: en/ru")
    parser.add_argument("-n", "--bot", type=str, default=None,
                        help="(n)name of the bot")
    parser.add_argument("-u", "--user", type=str, default=None,
                        help="your (u)username")
    args = parser.parse_args()
    # Starting the bot.
    talker = TalkWithMe(args)
    talker.main()
