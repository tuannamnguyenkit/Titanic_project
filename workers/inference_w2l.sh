MACHINE=i13hpc64
VGPU=$1

#FEATURES=$1
exec ssh $MACHINE '
#echo Language= '"$LANGUAGE"' >&2
setenv SYSTEM_PATH `dirname "$0"`
setenv WD `pwd`
setenv PYTHONPATH $SYSTEM_PATH/lib:/project/mt2020/project/namnguyen/titanic/Wav2Lip
setenv PATH "/home/mbehr/anaconda3/envs/lip/bin/:$PATH"
setenv CUDA_VISIBLE_DEVICES '"$VGPU"'
setenv CUDA_DEVICE_ORDER PCI_BUS_ID
echo '"$VGPU"'

#info_yaml=/project/OML/titanic/VoiceConv/pairs_iwslt_3.yaml
setenv video_path "/project/mt2020/project/namnguyen/titanic/Wav2Lip/waibel_30s.mp4|/project/mt2020/project/namnguyen/titanic/Wav2Lip/waibel_2.mp4|/project/mt2020/project/namnguyen/titanic/Wav2Lip/waibel_3.mp4|/project/mt2020/project/namnguyen/titanic/Wav2Lip/Hanselka_short.mp4|/project/mt2020/project/namnguyen/titanic/Wav2Lip/Stockton_short.mp4"
setenv name "AlexWaibel_1|AlexWaibel_2|AlexWaibel_3|Hanselka|Stockton"

#setenv  video_path "/project/mt2020/project/namnguyen/titanic/Wav2Lip/waibel_3.mp4|/project/mt2020/project/namnguyen/titanic/Wav2Lip/Stockton_short.mp4"
#setenv name "AlexWaibel|Stockton"

setenv audio_path /home/namnguyen/PycharmProjects/my_first_flask_app/audio/converted_gen.wav
setenv out_path /home/namnguyen/PycharmProjects/my_first_flask_app/audio
#setenv out_path /home/namnguyen/PycharmProjects/my_first_flask_app/audio/
#model_path=/project/OML/titanic/VoiceConv/model_german/model.ckpt-1000.pt
setenv model_path /project/mt2020/project/namnguyen/titanic/Wav2Lip/checkpoints/checkpoint_step000165000.pth

setenv pythonCMD "python -u -W ignore"
setenv WORKER "/home/namnguyen/PycharmProjects/my_first_flask_app/process/inference_w2l_stdin.py"

$pythonCMD $WORKER --checkpoint_path $model_path --name $name --face $video_path --audio $audio_path  --resize_all --read_from_folder --outpath $out_path --resize_factor 1   --wav2lip_batch_size 32 --face_det_batch_size 16
'
 
