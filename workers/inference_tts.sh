
MACHINE=i13hpc64
VGPU=$1

#FEATURES=$1
exec ssh $MACHINE  '
#echo Language= '"$LANGUAGE"' >&2
setenv SYSTEM_PATH `dirname "$0"`
setenv WD `pwd`
setenv PYTHONPATH $SYSTEM_PATH/lib:/project/mt2020/project/namnguyen/titanic/TTS/FastSpeech2
setenv PATH "/home/eugan/miniconda3/envs/titanic/bin/:$PATH"
setenv CUDA_VISIBLE_DEVICES '"$VGPU"'
setenv CUDA_DEVICE_ORDERPCI_BUS_ID
echo '"$MACHINE"'
echo '"$VGPU"'

setenv pythonCMD "python -u -W ignore"
setenv WORKER "/home/namnguyen/PycharmProjects/my_first_flask_app/process/inference_tts_stdin.py"
setenv out_path "/home/namnguyen/PycharmProjects/my_first_flask_app/audio/TTS_gen.wav"

setenv tts_restore_step 900000
setenv tts_p_config /export/data1/data/namnguyen/titanic/TTS/FastSpeech2/config/LJSpeech/preprocess.yaml
setenv tts_m_config /export/data1/data/namnguyen/titanic/TTS/FastSpeech2/config/LJSpeech/model.yaml
setenv tts_t_config /export/data1/data/namnguyen/titanic/TTS/FastSpeech2/config/LJSpeech/train.yaml

#$pythonCMD $WORKER --tts-restore_step $tts_restore_step --tts-mode '"stdin"' --tts-preprocess_config $tts_p_config --tts-model_config $tts_m_config --tts-train_config $tts_t_config --tts-duration_control 0.96
$pythonCMD $WORKER --restore_step $tts_restore_step --out_path $out_path --mode stdin --preprocess_config $tts_p_config --model_config $tts_m_config --train_config $tts_t_config --duration_control 1.2
'
