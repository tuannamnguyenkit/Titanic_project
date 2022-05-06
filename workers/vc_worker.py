import torch
import numpy as np

import soundfile as sf

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk

from parallel_wavegan.utils import load_model
from configs.vc_config import vc_config

import os
import time

from spectrogram import logmelspectrogram

import resampy
import pyworld as pw
from scipy.io import wavfile
import torchaudio

import yaml


def extract_logmel(audio, mean, std, adc=False, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    print(audio)
    if adc:
        wav, fs = sf.read(audio)
        print(wav.shape)
        print(wav.min(), wav.max(), wav.mean())
        if fs != sr:
            wav = resampy.resample(wav, fs, sr, axis=0)
            fs = sr
        assert fs == 16000
    else:
        wav = (torch.from_numpy(audio)).type(torch.FloatTensor)
        transform = torchaudio.transforms.Resample(22050, 16000)
        wav = transform(wav)
        wav = wav.numpy()
    # wavfile.write("/project/OML/titanic/VoiceConv/converted_iwslt_4/back_wav.wav",22050, wav)
    fs = sr
    # print(f"STDIN: {wav.shape}")
    # print(f"STDIN: {wav.min()},{wav.max()},{wav.mean()}")
    # wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    # assert fs == 16000
    peak = np.abs(wav).max()
    # print(wav.shape)
    if peak > 1.0:
        wav /= 32767.0
        # wavfile.write("/project/OML/titanic/VoiceConv/converted_iwslt_4/back.wav",16000, wav)
    # print(wav.min(), wav.max(), wav.mean())
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


class VC_worker:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
        self.encoder_lf0 = Encoder_lf0()
        self.encoder_spk = Encoder_spk()
        self.decoder = Decoder_ac(dim_neck=64)
        self.fp16 = args.fp16

        checkpoint_path = args.model_path
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        self.decoder.load_state_dict(checkpoint["decoder"])

        if self.fp16:
            self.encoder = self.encoder.half()
            self.decoder = self.decoder.half()
            self.encoder_spk = self.encoder_spk.half()


        self.encoder.to(self.device)
        self.encoder_lf0.to(self.device)
        self.encoder_spk.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.encoder_spk.eval()
        self.decoder.eval()

        mel_stats = np.load(args.mel_stat_path)
        self.mean = mel_stats[0]
        self.std = mel_stats[1]
        self.vocoder_checkpoint = args.vocoder_checkpoint
        self.outdir = args.outdir
        dirname = os.path.dirname(self.vocoder_checkpoint)
        config_path = os.path.join(dirname, "config.yml")
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        args = {'checkpoint': self.vocoder_checkpoint, 'outdir': self.outdir}
        config.update(args)
        print(f"Loaded model parameters from {self.vocoder_checkpoint}.")

        vocoder = load_model(self.vocoder_checkpoint, config)
        print("DDD")
        vocoder.remove_weight_norm()
        if self.fp16:
            vocoder = vocoder.half()
        self.vocoder = vocoder.eval().to(self.device)
        self.create_tgt_voice()

    def create_tgt_voice(self):
        list_wavpaths = self.args.tgtwav_path.split("|")
        self.list_name = self.args.name.split("|")
        list_ref_mel = []
        for name, facepath in zip(self.list_name, list_wavpaths):
            print(facepath)
            ref_mel, _ = extract_logmel(facepath, self.mean, self.std, adc=True)
            if self.fp16:
                ref_mel = torch.HalfTensor(ref_mel.T).unsqueeze(0).to(self.device)
            else:
                ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(self.device)

            list_ref_mel.append(ref_mel)
        self.list_ref_mel = list_ref_mel

    def inference(self, audio, speaker):

        if speaker in self.list_name:
            ref_mel = self.list_ref_mel[self.list_name.index(speaker)]
        else:
            print("Cannot find the name")
            ref_mel = self.list_ref_mel[0]

        src_mel, src_lf0 = extract_logmel(audio, self.mean, self.std, adc=False)
        if self.fp16:
            src_mel = torch.HalfTensor(src_mel.T).unsqueeze(0).to(self.device)
            src_lf0 = torch.HalfTensor(src_lf0).unsqueeze(0).to(self.device)
        else:
            src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(self.device)
            src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z, _, _, _ = self.encoder.encode(src_mel)
            lf0_embs = self.encoder_lf0(src_lf0)
            spk_emb = self.encoder_spk(ref_mel)
            output = self.decoder(z, lf0_embs, spk_emb)
            print(output)
        with torch.no_grad():

            c = output.clone().detach()

            c = c.squeeze(0).contiguous()

            y = self.vocoder.inference(c, normalize_before=False, fp16=self.fp16).view(-1).cpu().numpy()

        return y


if __name__ == '__main__':
    print("testing")
    print(vc_config.mel_stat_path)
    model = VC_worker(vc_config)
    wav, fs = sf.read("/home/titanic/PycharmProjects/Titanic_project/audio/tts.wav")
    print(fs)
    start = time.time()
    converted_wav = model.inference(wav, "AlexWaibel")
    end = time.time()
    print(end - start)
    # print(converted_wav.dtype)
    wavfile.write("audio/converted_wav", 16000, np.float32(converted_wav))

    time.sleep(5)