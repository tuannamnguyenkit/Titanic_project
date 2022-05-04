import torch
import numpy as np

import soundfile as sf

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk

from parallel_wavegan.bin import decode_pipe
from parallel_wavegan.utils import load_model

import os
import time
import sys
import base64

import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw
from scipy.io import wavfile
import torchaudio

import argparse
import yaml

voiceconv_err_file = open('/project/mt2020/project/namnguyen/titanic/VoiceConv/err_log.log', 'a+')


def extract_logmel(wav_path, mean, std, adc=None, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    if not adc:
        wav, fs = sf.read(wav_path)
        print(wav.shape)
        print(wav.min(), wav.max(), wav.mean())
        if fs != sr:
            wav = resampy.resample(wav, fs, sr, axis=0)
            fs = sr
        assert fs == 16000
    else:
        wav_ = np.frombuffer(base64.b64decode(adc), dtype="int16")
        wav = (torch.from_numpy(wav_)).type(torch.FloatTensor)
        transform = torchaudio.transforms.Resample(22050, 16000)
        wav = transform(wav)
        wav = wav.numpy()
        # wavfile.write("/project/OML/titanic/VoiceConv/converted_iwslt_4/back_wav.wav",22050, wav)
        fs = sr
        print(f"STDIN: {wav.shape}")
        print(f"STDIN: {wav.min()},{wav.max()},{wav.mean()}")
    # wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    # assert fs == 16000
    peak = np.abs(wav).max()
    print(wav.shape)
    if peak > 1.0:
        wav /= 32767.0
        # wavfile.write("/project/OML/titanic/VoiceConv/converted_iwslt_4/back.wav",16000, wav)
    print(wav.min(), wav.max(), wav.mean())
    mel = logmelspectrogram(
        x=wav,
        fs=fs,
        n_mels=80,
        n_fft=400,
        n_shift=160,
        win_length=400,
        window='hann',
        fmin=80,
        fmax=7600,
    )

    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160 / fs * 1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])  # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0


def convert(args):
    # src_wav_path = args.source_wav
    # ref_wav_path = args.reference_wav

    # load voice conversion model
    # out_dir = args.converted_wav_path
    # os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    encoder_lf0 = Encoder_lf0()
    encoder_spk = Encoder_spk()
    decoder = Decoder_ac(dim_neck=64)
    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    decoder.to(device)

    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_spk.eval()
    decoder.eval()
    print("1")
    mel_stats = np.load('/project/mt2020/project/namnguyen/titanic/VoiceConv/mel_stats/stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    list_facepaths = args.tgtwav_path.split("|")
    list_name = args.name.split("|")
    list_ref_mel = []
    for name, facepath in zip(list_name, list_facepaths):
        ref_mel, _ = extract_logmel(facepath, mean, std)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        list_ref_mel.append(ref_mel)

    # load vocoder
    chkp = "/project/mt2020/project/namnguyen/titanic/VoiceConv/vocoder/checkpoint-3000000steps.pkl"
    outdir = "/project/mt2020/project/namnguyen/titanic/VoiceConv/converted_iwslt_4"

    dirname = os.path.dirname(chkp)
    config = os.path.join(dirname, "config.yml")
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    converted_wav_path = args.converted_wav_path
    args = {'checkpoint': chkp, 'outdir': outdir}
    config.update(args)
    print(f"Loaded model parameters from {chkp}.")

    vocoder = load_model(chkp, config)
    print("DDD")
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    i = 0

    # print(infos)
    print("start")
    print(list_name)
    print("VoiceConv READY")
    for line in sys.stdin:
        i += 1
        # src_wav_path = pair["source"]
        if not line:
            continue

        adc, name = line.strip().split("\t")
        print(name)
        if adc == "":
            print("empty signal", file=sys.stderr)
            continue
        if name in list_name:
            ref_mel = list_ref_mel[list_name.index(name)]
        else:
            print("Cannot find the name")
            ref_mel = list_ref_mel[0]
        start = time.time()
        src_mel, src_lf0 = extract_logmel("adc", mean, std, adc=adc.strip())
        src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
        src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)

        with torch.no_grad():
            z, _, _, _ = encoder.encode(src_mel)
            lf0_embs = encoder_lf0(src_lf0)
            spk_emb = encoder_spk(ref_mel)
            output = decoder(z, lf0_embs, spk_emb)

        print("Time for voice conversion")
        print(time.time() - start)
        print('synthesize waveform...')
        start = time.time()
        print(output.shape)

        with torch.no_grad():

            c = torch.tensor(output, dtype=torch.float).to(device)
            c = c.squeeze(0).contiguous()
            y = vocoder.inference(c, normalize_before=False).view(-1).cpu().numpy()
            print("Time for vocoder")
            print(time.time() - start)
            print(f"STDOUT: {y.shape}")
            print(f"STDOUT: {y.min()},{y.max()},{y.mean()}")
            sf.write(
                converted_wav_path,
                y,
                config["sampling_rate"],
                "PCM_16",
            )

            res = base64.b64encode(y).decode('latin-1')

            sys.stdout.write(f"ADC:{res} \n")
            sys.stdout.flush()

        # cmd = ['/project/OML/titanic/VoiceConv/parallel-wavegan-decode.sh']
        # asr_de_proc = subprocess.Popen('/project/OML/titanic/VoiceConv/parallel-wavegan-decode.sh',
        #                            shell=True, encoding="utf-8", bufsize=0, universal_newlines=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=voiceconv_err_file)
        # for line in asr_de_proc.stdout:
        #    res = line
        #    print(res)


#  cmd = ['parallel-wavegan-decode', '--checkpoint', \
#          './vocoder/checkpoint-3000000steps.pkl', \
#        '--feats-scp', f'{str(out_dir)}/feats.scp', '--outdir', str(out_dir)]
# subprocess.call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgtwav_path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    # parser.add_argument("--info_path", "-i", type=str, required=True)
    parser.add_argument('--converted_wav_path', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    args = parser.parse_args()
    print(args.converted_wav_path)
    convert(args)
