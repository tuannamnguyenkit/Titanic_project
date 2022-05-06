from . import Struct as W2L_Config

w2l_config = W2L_Config("W2L_Config")

w2l_config.checkpoint_path = "/home/titanic/titanic/Wav2Lip/checkpoints/checkpoint_step000165000.pth"
w2l_config.face = "/home/titanic/titanic/Wav2Lip/waibel_30s.mp4|/home/titanic/titanic/Wav2Lip/waibel_2.mp4|/home/titanic/titanic/Wav2Lip/waibel_3.mp4|/home/titanic/titanic/Wav2Lip/Hanselka_short.mp4|/home/titanic/titanic/Wav2Lip/Stockton_short.mp4"
w2l_config.name = "AlexWaibel_1|AlexWaibel_2|AlexWaibel_3|Hanselka|Stockton"
w2l_config.audio = "audio/converted_wav"
w2l_config.outpath = "audio"
w2l_config.static = False

w2l_config.fps = 25
w2l_config.pads = [0, 10, 0, 0]
w2l_config.face_det_batch_size = 16
w2l_config.wav2lip_batch_size = 16
w2l_config.resize_factor = 1
w2l_config.crop = [0, -1, 0, -1]
w2l_config.box = [-1, -1, -1, -1]
w2l_config.rotate = False
w2l_config.nosmooth = False
w2l_config.read_from_folder = True
w2l_config.resize_all = True
w2l_config.img_size = 96
w2l_config.fp16 = False
