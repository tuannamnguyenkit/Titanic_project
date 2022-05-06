from . import Struct as TTS_Config

tts_config = TTS_Config("TTS_Config")

tts_config.restore_step = 900000

tts_config.source = None
tts_config.text = None
tts_config.speaker_id = 0

tts_config.preprocess_config = "/home/titanic/titanic/TTS/FastSpeech2/config/LJSpeech/preprocess.yaml"
tts_config.model_config = "/home/titanic/titanic/TTS/FastSpeech2/config/LJSpeech/model.yaml"
tts_config.train_config = "/home/titanic/titanic/TTS/FastSpeech2/config/LJSpeech/train.yaml"
tts_config.pitch_control = 1.0
tts_config.energy_control = 1.0
tts_config.duration_control = 1.0
tts_config.pitch_control_word_level = []
tts_config.energy_control_word_level = []
tts_config.duration_control_word_level = []
tts_config.fp16 = True
tts_config.xml_input = False
