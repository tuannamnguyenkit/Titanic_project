from . import Struct as TTS_Config

tts_config = TTS_Config("TTS_Config")

tts_config.restore_step = 90000

tts_config.source = None
tts_config.text = None
tts_config.speaker_id = 0

tts_config.preprocess_config = "path/to/preprocess.yaml"
tts_config.model_config = "path/to/train.yaml"

tts_config.pitch_control = 1.0
tts_config.energy_control = 1.0
tts_config.duration_control = 1.0
tts_config.pitch_control_word_level = []
tts_config.energy_control_word_level = []

tts_config.xml_input = False
