## A simple voice chatbot program running from terminal

*Tested on Gentoo GNU/Linux and Python 3.9. Supported languages: English and Russian.*

#### You might like this program if:

 - you want to practice your English or Russian speaking skills
 - you are forever alone (as I am) and have nobody to talk with
 - you're planing to create voice assistant from the ground

#### Used components:

 - [Sber's ruGPT-3](https://github.com/ai-forever/ru-gpts "ai-forever/ru-gpts") as dialog generation system
 - [OpenAI's Whisper](https://github.com/openai/whisper "openai/whisper") as speech to text
 - [RHVoice](https://github.com/RHVoice/RHVoice "RHVoice/RHVoice") or [silero-models](https://github.com/snakers4/silero-models "snakers4/silero-models") as text to speech
 - [WebrtcVAD](https://github.com/wiseman/py-webrtcvad "wiseman/py-webrtcvad") for voice activity detection;
 - **sounddevice** for stream catching and **simpleaudio** for playing generated audio

#### Running the program:

`$ python talk_with_me.py --stt tiny --gpt small --lang en --tts silero --sex female --bot Helen --user Harry`

or

`$ ./talk_with_me.py -r base -b medium -l ru -s rhvoice -g male -n Хер\ с\ горы -u Весёлый\ мясник`

See `talk_with_me.py --help` for more details. Remember, quality of speech recognition and text generation depends on model size, so you have to get all models into VRAM. For examle, on my Nvidia GTX 1060 6Ĝb it fits well with `-r base -b large` and `-r medium -b medium`. If you have good micrphone you can save memory using **tiny** or **small** whisper model to use larger model for text generation. On the other side, having a laptop with integrated mic it's more suitable to use **medium** whisper model.

If you want to use **RHVoice** you probably have (at least on Windows) to run `pip install rhvoice-wrapper-bin`.

Linux users can install it globally, on Gentoo run those commands:

`# layman -a guru`

`# echo "app-accessibility/rhvoice ~amd64" >> /etc/portage/ package.accept_keywords/apps`

`# emerge -av rhvoice`

or check instructions for your distribution on the Internet.

**Silero-models** have some issues (at least in my case) with pytorch>=1.11.0, for this reason used python 3.9 and pytorch 1.10.2. If you don't want to use silero-models, but rhvoice instead, python 3.10 works fine too.
