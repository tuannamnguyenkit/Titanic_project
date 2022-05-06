from . import Struct as VC_Config

vc_config = VC_Config("VC_Config")

vc_config.model_path = "/home/titanic/titanic/VoiceConv/model_en_waibel/model.en_Alex.4.pt"
vc_config.vocoder_checkpoint = "/home/titanic/titanic/VoiceConv/vocoder/checkpoint-3000000steps.pkl"
vc_config.fp16 = True

vc_config.tgtwav_path = "/home/titanic/titanic/Alex_audio/subtask_1-usr0051_10.wav|/home/titanic/titanic/HanSelka_audio/HanSelka_out.wav|/home/titanic/titanic/Stockton_audio/4160_14187_000041_000000-Srush.wav"
vc_config.name = "AlexWaibel|Hanselka|Stockton"
vc_config.mel_stat_path = "/home/titanic/titanic/VoiceConv/VQMIVC/mel_stats/stats.npy"
vc_config.outdir = "/home/titanic/titanic/VoiceConv/converted_iwslt_4"
vc_config.converted_wav_path = "converted_gen.wav"
