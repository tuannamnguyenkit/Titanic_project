from . import Struct as W2L_Config

w2l_config = W2L_Config("W2L_Config")

w2l_config.checkpoint_path = "path/model"
w2l_config.face = "|"
w2l_config.name = "|"
w2l_config.audio = "audio/converted_gen.wav"
w2l_config.outpath = "audio/Alex.mp4"
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
w2l_config.read_from_folder = False
w2l_config.resize_all = False
