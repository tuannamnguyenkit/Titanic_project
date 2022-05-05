MACHINE=i13hpc64
VGPU=$1

#FEATURES=$1
exec ssh $MACHINE '
#echo Language= '"$LANGUAGE"' >&2
setenv SYSTEM_PATH `dirname "$0"`
setenv WD `pwd`
#export PYTHONPATH=$SYSTEM_PATH/lib:/project/OML/titanic/TTS/FastSpeech2/
setenv PYTHONPATH $SYSTEM_PATH/lib:/project/mt2020/project/namnguyen/titanic/VoiceConv/VQMIVC/ParallelWaveGAN/parallel_wavegan/
setenv PYTHONPATH $SYSTEM_PATH/lib:/project/mt2020/project/namnguyen/titanic/VoiceConv/VQMIVC/ParallelWaveGAN/:/project/mt2020/project/namnguyen/titanic/VoiceConv/VQMIVC
setenv PATH "/home/eugan/miniconda3/envs/titanic/bin/:$PATH"
setenv CUDA_VISIBLE_DEVICES '"$VGPU"'
setenv CUDA_DEVICE_ORDER PCI_BUS_ID
echo '"$VGPU"'

#info_yaml=/project/OML/titanic/VoiceConv/pairs_iwslt_3.yaml
setenv tgtwav_path "/project/mt2020/project/namnguyen/titanic/Alex_audio/subtask_1-usr0051_10.wav|/project/mt2020/project/namnguyen/titanic/HanSelka_audio/HanSelka_out.wav|/project/mt2020/project/namnguyen/titanic/Stockton_audio/4160_14187_000041_000000-Srush.wav"
setenv name "AlexWaibel|Hanselka|Stockton"
setenv out_path /home/namnguyen/PycharmProjects/my_first_flask_app/audio/converted_gen.wav
#model_path=/project/OML/titanic/VoiceConv/model_german/model.ckpt-1000.pt
setenv model_path /project/mt2020/project/namnguyen/titanic/VoiceConv/model_en_waibel/model.en_Alex.4.pt

setenv pythonCMD "python -u -W ignore"
setenv WORKER "/home/namnguyen/PycharmProjects/my_first_flask_app/process/inference_vc_stdin.py"
$pythonCMD $WORKER --tgtwav_path $tgtwav_path --name $name --converted_wav_path $out_path -m $model_path
'
 
