import re
import argparse
from string import punctuation
import sys
import uuid
import base64

print(sys.path)
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import montreal_forced_aligner.g2p.generator as g2p_generator
from scipy.io import wavfile

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
from sys import stdin, exit, stdout
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def pitch_control_word_to_phoneme(pitch_control_word_level, text):
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


def energy_control_word_to_phoneme(energy_control_word_level, text):
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


def duration_control_word_to_phoneme(duration_control_word_level, text):
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


# Limits: No error checking,
#         Additional tags INSIDE emphasis tag will cause weird behaviour!
#         Only works with procentual parameters! (e.g. <prosody rate=150%>)
def parse_xml_input(xml_text):
    cleaned_text = []
    pitch_control_word_level = []
    energy_control_word_level = []
    duration_control_word_level = []

    current_pitch = []
    current_energy = []
    current_duration = []

    xml_text = xml_text.replace("<prosody rate", "<prosody-rate").replace("<prosody volume", "<prosody-volume").replace(
        "<prosody range", "<prosody-range")
    xml_text = xml_text.replace("</prosody rate", "</prosody-rate").replace("</prosody volume",
                                                                            "</prosody-volume").replace(
        "</prosody range", "</prosody-range")

    # TODO: check for syntax errors!
    text_parts = re.split(r'[ >]', xml_text)
    # filter empty strings
    text_parts = list(filter(None, text_parts))

    print("text parts: " + str(text_parts))

    for text in text_parts:
        if text[0] == '<':
            # tag
            if text[1] != '/':
                print("opening tag")
                # opening tag
                to_filter = '"% '
                if text[1:15] == 'prosody-range=':
                    print("pitch")
                    # pitch
                    argument = text[15:].translate({ord(i): None for i in to_filter})
                    # procentual value to multiplier
                    argument = float(argument) / 100.0
                    current_pitch.append(argument)
                elif text[1:16] == 'prosody-volume=':
                    print("energy")
                    # energy
                    argument = text[16:].translate({ord(i): None for i in to_filter})
                    argument = float(argument) / 100.0
                    current_energy.append(argument)
                elif text[1:14] == 'prosody-rate=':
                    print("duration")
                    # duration
                    argument = text[14:].translate({ord(i): None for i in to_filter})
                    argument = float(argument) / 100.0
                    # Higher prosody rate tag means faster speech, but internally duration parameter is handled the opposite way => use inverse
                    argument = 1.0 / argument
                    current_duration.append(argument)
                elif text[1:9] == 'emphasis':
                    # emphasis
                    print("emph")
                    current_duration.append(0.85)
                    current_pitch.append("HIGH")
                else:
                    print("Warning! Unknown tag: {}".format(text))
            else:
                print("closing tag")
                if text[2:16] == 'prosody-range':
                    print("pitch")
                    # pitch
                    current_pitch.pop()
                elif text[2:17] == 'prosody-volume':
                    print("energy")
                    # energy
                    current_energy.pop()
                elif text[2:15] == 'prosody-rate':
                    print("duration")
                    # duration
                    current_duration.pop()
                elif text[2:10] == 'emphasis':
                    # emphasis
                    print("close emph")
                    current_duration.pop()
                    current_pitch.pop()
                else:
                    print("Warning! Unknown tag: {}".format(text))
                    # closing tag

        else:
            # text to synthesize
            cleaned_text.append(text)
            text_pitch = 1.0 if not current_pitch else current_pitch[-1]
            text_energy = 1.0 if not current_energy else current_energy[-1]
            text_duration = 1.0 if not current_duration else current_duration[-1]
            pitch_control_word_level.append(text_pitch)
            energy_control_word_level.append(text_energy)
            duration_control_word_level.append(text_duration)

    cleaned_text = ' '.join(cleaned_text)
    print("Cleaned text: " + cleaned_text)
    print("pitch control: " + str(pitch_control_word_level))
    print("energy control: " + str(energy_control_word_level))
    print("duration control: " + str(duration_control_word_level))

    return cleaned_text, pitch_control_word_level, energy_control_word_level, duration_control_word_level


def synthesize(model, step, configs, vocoder, batchs, control_values, out_path):
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
            wavfile.write(out_path, sampling_rate, wav)
            base64_wav = base64.b64encode(wav)
            str_wav = base64_wav.decode('latin-1')
            # print(34343)
            full_time = time.time()
            print("mel_time: {}, wav_time {}, full_time: {}".format(mel_time - start_time, full_time - mel_time,
                                                                    full_time - start_time))
            sys.stdout.write(f"ADC:{str_wav} \n")
            sys.stdout.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single", "stdin"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="path to a output wav file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--pitch_control_word_level",
        nargs='*',
        action='append',
        default=[],
        help="control the pitch of the utterance word by word. One float value per word. Larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control_word_level",
        type=float,
        nargs='*',
        action='append',
        default=[],
        help="control the energy of the utterance word by word. One float value per word. Larger value for higher energy",
    )
    parser.add_argument(
        "--duration_control_word_level",
        type=float,
        nargs='*',
        action='append',
        default=[],
        help="control the speed of the utterance word by word. One float value per word. Larger value for slower speaking rate",
    )
    parser.add_argument(
        "--xml_input",
        action="store_true",
        help="If this flag is set, SSML Tags are parsed. Currently supported tags: <emphasis>, <prosody rate>, <prosody volume>, <prosody rage>. Will overwrite all given control parameters")
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Load G2P model
    # TODO: add error handling if g2p model is not found and get paths/phones from config
    # g2p = g2p_generator.PyniniWordListGenerator("lexicon/empty.txt", g2p_model_path="lexicon/german_prosodylab_g2p.zip")
    # g2p = g2p_generator.PyniniWordListGenerator("/project/OML/titanic/TTS/FastSpeech2/lexicon/empty.txt",
    #                                           g2p_model_path="/project/OML/titanic/TTS/FastSpeech2/lexicon/german_prosodylab_g2p.zip")
    # g2p.setup()
    legal_phones = ['b', 'E0', 'y0', 'a1', 'U0', '+', 'B1', 't', 'I0', 'i1', 'h', '/1', 'g', 'O1', 'd', 'q1', 'q0',
                    '&1', 'k', '|0', 'j', '=', 'B0', 'v', 'o1', 'a0', 'Y0', 'z', 'Y1', 'e1', '&0', 'E1', '~1', 's',
                    'e0', 'Z', '/0', 'l', 'm', 'W0', 'I1', 'r', 'S', 'x', 'i0', 'X0', 'p', '@0', 'n', 'f', 'O0', 'y1',
                    'null1', 'U1', 'X1', 'u0', 'J', ')1', ')0', 'N', '|1', 'u1', 'o0', 'W1']

    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    #
    g2p = G2p()

    if args.mode == "stdin":
        speakers = np.array([args.speaker_id])
        cache_dict = {}
        print("Up and running! Waiting for inputs on stdin...")
        try:
            for line in stdin:
                print(line)
                input_txt = line.rstrip()
                if input_txt in [None, ""]:
                    continue
                # TODO: Why 100 char limit?

                # get pitch control

                if args.pitch_control_word_level != []:
                    print(args.pitch_control_word_level[0])
                    pitch_control_phoneme_level = pitch_control_word_to_phoneme(args.pitch_control_word_level[0],
                                                                                args.text)
                    pitch_control = pitch_control_phoneme_level
                else:
                    pitch_control = args.pitch_control
                # get energy control
                if args.energy_control_word_level != []:
                    print(args.energy_control_word_level[0])
                    energy_control_phoneme_level = energy_control_word_to_phoneme(args.energy_control_word_level[0],
                                                                                  args.text)
                    energy_control = energy_control_phoneme_level
                else:
                    energy_control = args.energy_control
                # get duration control
                if args.duration_control_word_level != []:
                    print(args.duration_control_word_level[0])
                    duration_control_phoneme_level = duration_control_word_to_phoneme(
                        args.duration_control_word_level[0], args.text)
                    duration_control = duration_control_phoneme_level
                else:
                    duration_control = args.duration_control
                control_values = pitch_control, energy_control, duration_control

                ids = [str(uuid.uuid4())]
                cache_dict[input_txt] = ids[0]
                raw_texts = [input_txt[:100]]

                texts = np.array([preprocess_text(input_txt, preprocess_config, lexicon, g2p)])

                text_lens = np.array([len(texts[0])])
                batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
               # print("RES: {}.wav".format(ids[0]))
                synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, args.out_path)

        except KeyboardInterrupt:
            print("Recieved SIGINT, exiting...")
            exit(0)
    if args.xml_input:
        text, pitch_control_word_level, energy_control_word_level, duration_control_word_level = parse_xml_input(
            args.text)
        pitch_control_phoneme_level = pitch_control_word_to_phoneme(pitch_control_word_level, text)
        energy_control_phoneme_level = energy_control_word_to_phoneme(energy_control_word_level, text)
        duration_control_phoneme_level = duration_control_word_to_phoneme(duration_control_word_level, text)
        control_values = pitch_control_phoneme_level, energy_control_phoneme_level, duration_control_phoneme_level
    else:
        # get control values
        text = args.text
        if args.pitch_control_word_level != []:
            print(args.pitch_control_word_level[0])
            pitch_control_phoneme_level = pitch_control_word_to_phoneme(args.pitch_control_word_level[0], args.text)
            pitch_control = pitch_control_phoneme_level
        else:
            pitch_control = args.pitch_control

        if args.energy_control_word_level != []:
            print(args.energy_control_word_level[0])
            energy_control_phoneme_level = energy_control_word_to_phoneme(args.energy_control_word_level[0], args.text)
            energy_control = energy_control_phoneme_level
        else:
            energy_control = args.energy_control

        if args.duration_control_word_level != []:
            print(args.duration_control_word_level[0])
            duration_control_phoneme_level = duration_control_word_to_phoneme(args.duration_control_word_level[0],
                                                                              args.text)
            duration_control = duration_control_phoneme_level
        else:
            duration_control = args.duration_control
        control_values = pitch_control, energy_control, duration_control

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "de":
            texts = np.array([preprocess_german(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
