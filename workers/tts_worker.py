import re
import argparse
from string import punctuation
import sys
import uuid
import base64
from configs.tts_config import tts_config

print(sys.path)
import torch
import yaml
import numpy as np
# from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from scipy.io import wavfile

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
# from dataset import TextDataset
from text import text_to_sequence
# from sys import stdin, exit, stdout
import time


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_text(text, preprocess_config, lexicon=None, g2p=None):
    if preprocess_config["dataset"] == "LJSpeech":
        return preprocess_english(text, preprocess_config, lexicon, g2p)
    else:
        return preprocess_german(text, preprocess_config)


def preprocess_english(text, preprocess_config, lexicon, g2p):
    text = text.rstrip(punctuation)

    phones = []
    words = [word for word in re.split(r"([,;.\-\?\!\s+])", text) if word != " "]
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            print("Word not found in dict: {}".format(w))
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = phones.replace("{=}", "{Z0}")
    print("PHONES =================== {}".format(str(phones)))
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("{Z0}", "{=}")
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    print("len phones: {}".format(len(sequence)))
    return np.array(sequence)


def preprocess_german(text, preprocess_config):
    text = text.rstrip(punctuation)
    text = text.replace("ß", "ss")
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        w = w.lower()
        if w in lexicon:
            phones += lexicon[w]
        else:
            if w in ["", " ", ","]:
                continue
            if w == ".":
                phones += "{sp}"
                continue
            print("Word not found in dict: {}".format(w))
            g2p.word_list = [w]
            pronunciations = g2p.generate_pronunciations()[w].pronunciations
            pronunciation = pronunciations.pop().pronunciation
            # TODO: Check if phones in pronunciation are in phone set!
            if not (set(pronunciation).issubset(legal_phones)):
                print("FOUND ILEGAL PHONE: " + str(set(pronunciation).difference(set(legal_phones))))
            phones += pronunciation
    phones = "{" + "}{".join(phones) + "}"
    phones = phones.replace("{=}", "{Z0}")
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("{Z0}", "{=}")
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    print("len phones: {}".format(len(sequence)))
    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def pitch_control_word_to_phoneme(pitch_control_word_level, text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    pitch_control_phoneme_level = []
    g2p = G2p()

    words = re.split(r"([,;.\-\?\!\s+])", text)
    proper_words = [word for word in words if re.search('[a-zA-Z]', word) is not None]
    if len(proper_words) != len(pitch_control_word_level):
        print("Word amount and word level pitch control parameter amount does not match!")
        print("pitch control parameters amount: {} (parameters: {})".format(len(pitch_control_word_level),
                                                                            pitch_control_word_level))
        print("word amount: {} (words: {})".format(len(proper_words), proper_words))
        return 1.0
    proper_word_index = 0
    print(proper_words)

    for w in words:
        if w.lower() in lexicon:
            phone_amount = len(lexicon[w.lower()])
        else:
            print("Word not found in dict: {}".format(w))
            phone_amount = len(list(filter(lambda p: p != " ", g2p(w))))
        pitch_control_phoneme_level += [pitch_control_word_level[proper_word_index]] * phone_amount
        if w.lower() == proper_words[proper_word_index].lower():
            proper_word_index += 1
            if proper_word_index >= len(proper_words):
                break
    if len(proper_words) != proper_word_index:
        print("Bug Warnign! len proper words: {}, proper_word_index: {}".format(len(proper_words), proper_word_index))

    print("PCPL: {}".format(pitch_control_phoneme_level))
    if text[-1] == " ":
        pitch_control_phoneme_level += [1.0]
    return pitch_control_phoneme_level


def energy_control_word_to_phoneme(energy_control_word_level, text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    energy_control_phoneme_level = []
    g2p = G2p()

    words = re.split(r"([,;.\-\?\!\s+])", text)
    proper_words = [word for word in words if re.search('[a-zA-Z]', word) is not None]
    if len(proper_words) != len(energy_control_word_level):
        print("Word amount and word level energy control parameter amount does not match!")
        print("energy control parameters amount: {} (parameters: {})".format(len(energy_control_word_level),
                                                                             energy_control_word_level))
        print("word amount: {} (words: {})".format(len(proper_words), proper_words))
        return 1.0
    proper_word_index = 0
    print(proper_words)

    for w in words:
        if w.lower() in lexicon:
            phone_amount = len(lexicon[w.lower()])
        else:
            print("Word not found in dict: {}".format(w))
            phone_amount = len(list(filter(lambda p: p != " ", g2p(w))))
        energy_control_phoneme_level += [energy_control_word_level[proper_word_index]] * phone_amount
        if w.lower() == proper_words[proper_word_index].lower():
            proper_word_index += 1
            if proper_word_index >= len(proper_words):
                break
    if len(proper_words) != proper_word_index:
        print("Bug Warning! len proper words: {}, proper_word_index: {}".format(len(proper_words), proper_word_index))

    print("ECPL: {}".format(energy_control_phoneme_level))
    if text[-1] == " ":
        energy_control_phoneme_level += [1.0]
    return energy_control_phoneme_level


def duration_control_word_to_phoneme(duration_control_word_level, text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    duration_control_phoneme_level = []
    g2p = G2p()

    words = re.split(r"([,;.\-\?\!\s+])", text)
    proper_words = [word for word in words if re.search('[a-zA-Z]', word) is not None]
    if len(proper_words) != len(duration_control_word_level):
        print("Word amount and word level duration control parameter amount does not match!")
        print("duration control parameters amount: {} (parameters: {})".format(len(duration_control_word_level),
                                                                               duration_control_word_level))
        print("word amount: {} (words: {})".format(len(proper_words), proper_words))
        return 1.0
    proper_word_index = 0
    print(proper_words)

    for w in words:
        if w.lower() in lexicon:
            phone_amount = len(lexicon[w.lower()])
        else:
            print("Word not found in dict: {}".format(w))
            phone_amount = len(list(filter(lambda p: p != " ", g2p(w))))
        duration_control_phoneme_level += [duration_control_word_level[proper_word_index]] * phone_amount
        if w.lower() == proper_words[proper_word_index].lower():
            print(w)
            proper_word_index += 1
            if proper_word_index >= len(proper_words):
                break
    if len(proper_words) != proper_word_index:
        print("Bug Warning! len proper words: {}, proper_word_index: {}".format(len(proper_words), proper_word_index))

    print("DCPL: {}".format(duration_control_phoneme_level))
    if text[-1] == " ":
        duration_control_phoneme_level += [1.0]
    return duration_control_phoneme_level


def synthesize(model, configs, vocoder, batchs, control_values, device):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            start_time = time.time()
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            mel_time = time.time()
            wav_predictions = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            wav = wav_predictions[0]
            print(wav.shape)
            print(wav.min(), wav.max(), wav.mean())

            # wavfile.write(out_path, sampling_rate, wav)
            # base64_wav = base64.b64encode(wav)
            # str_wav = base64_wav.decode('latin-1')
            # print(34343)
            full_time = time.time()
            print("mel_time: {}, wav_time {}, full_time: {}".format(mel_time - start_time, full_time - mel_time,
                                                                    full_time - start_time))

            return wav
        # sys.stdout.write(f"ADC:{str_wav} \n")

        # sys.stdout.flush()


class TTS_worker:
    def __init__(self, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.speakers = np.array([args.speaker_id])
        self.preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        self.model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        self.train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
        self.configs = (self.preprocess_config, self.model_config, self.train_config)
        self.sampling_rate = self.preprocess_config["preprocessing"]["audio"]["sampling_rate"]

        self.model = get_model(args, self.configs, self.device, train=False, fp16=args.fp16)
        self.vocoder = get_vocoder(self.model_config, self.device, fp16=args.fp16)
        #  print("half")
        self.legal_phones = ['b', 'E0', 'y0', 'a1', 'U0', '+', 'B1', 't', 'I0', 'i1', 'h', '/1', 'g', 'O1', 'd', 'q1',
                             'q0',
                             '&1', 'k', '|0', 'j', '=', 'B0', 'v', 'o1', 'a0', 'Y0', 'z', 'Y1', 'e1', '&0', 'E1', '~1',
                             's',
                             'e0', 'Z', '/0', 'l', 'm', 'W0', 'I1', 'r', 'S', 'x', 'i0', 'X0', 'p', '@0', 'n', 'f',
                             'O0',
                             'y1',
                             'null1', 'U1', 'X1', 'u0', 'J', ')1', ')0', 'N', '|1', 'u1', 'o0', 'W1']
        self.lexicon = read_lexicon(self.preprocess_config["path"]["lexicon_path"])
        self.g2p = G2p()

    def inference(self, input_txt):
        if input_txt in [None, ""]:
            return None
        if self.args.pitch_control_word_level != []:
            print(self.args.pitch_control_word_level[0])
            pitch_control_phoneme_level = pitch_control_word_to_phoneme(self.args.pitch_control_word_level[0],
                                                                        self.args.text, self.preprocess_config)
            pitch_control = pitch_control_phoneme_level
        else:
            pitch_control = self.args.pitch_control
            # get energy control
        if self.args.energy_control_word_level != []:
            print(self.args.energy_control_word_level[0])
            energy_control_phoneme_level = energy_control_word_to_phoneme(self.args.energy_control_word_level[0],
                                                                          self.args.text)
            energy_control = energy_control_phoneme_level
        else:
            energy_control = self.args.energy_control
            # get duration control
        if self.args.duration_control_word_level != []:
            print(self.args.duration_control_word_level[0])
            duration_control_phoneme_level = duration_control_word_to_phoneme(
                self.args.duration_control_word_level[0], self.args.text)
            duration_control = duration_control_phoneme_level
        else:
            duration_control = self.args.duration_control
        control_values = pitch_control, energy_control, duration_control
        raw_texts = [input_txt[:100]]
        texts = np.array([preprocess_text(input_txt, self.preprocess_config, self.lexicon, self.g2p)])
        text_lens = np.array([len(texts[0])])
        batchs = [("333", raw_texts, self.speakers, texts, text_lens, max(text_lens))]
        wav = synthesize(self.model, self.configs, self.vocoder, batchs, control_values, self.device)
        return wav


import time

if __name__ == '__main__':
    print("testing")
    tts_worker = TTS_worker(tts_config)
    wav = tts_worker.inference(input_txt="How are you today")
    #
    # wav = tts_worker.inference(input_txt="one two three four five six seven eight")
    wavfile.write("audio/tts.wav", tts_worker.sampling_rate, wav)
# torch.cuda.empty_cache()
