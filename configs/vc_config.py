from . import Struct as VC_Config

vc_config = VC_Config("VC_Config")

vc_config.model_path = "path/to/model"
vc_config.vocoder_checkpoint = "path/to/vocoder"

vc_config.tgtwav_path ="|"
vc_config.name = "|"
vc_config.converted_wav_path = "converted_gen.wav"