from flask import Flask, render_template, Response, request, redirect, url_for
from threading import Thread, Event
from flask_socketio import SocketIO, emit
import cv2
import os
import subprocess
import shutil
import time
import numpy as np
import base64
import argparse
import uuid
# create the flask app
import threading

sLock = threading.Semaphore()

app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)

thread = Thread()
thread_stop_event = Event()

list_video_paths = []
speaker_list = ["AlexWaibel","Stockton"]

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


# what html should be loaded as the home page when the app loads?

@app.route('/')
def index():
    return redirect(url_for('home'))


@app.route('/home')
def home():
    return render_template('simple_template_2.html')


@app.route('/demo')
def demo():
    return render_template('simple_template_demo.html')

@app.route('/add', methods=['GET'])
def add():
    list_video_paths.append(["AlexWaibel_1_321_384.mp4","1","Hello"])
    return "0"


@app.route('/video_generate', methods=['GET', 'POST'])
def video_generate():
    sLock.acquire()
    try:
        content = request.json
        message = content['text']
        name = content['speaker']
        message = message.strip() + "\n"
        start = time.time()
        try:
            tts_en_proc.stdin.write(message)
        except Exception as e:
            print(f"E: {e}")
        print("!1111!!")
        tts_en_proc.stdin.flush()
        print("ASD")
        tts_wav = ""
        if name in speaker_list:
            speaker_idx = speaker_list.index(name)
        else:
            speaker_idx = 0
        for line in tts_en_proc.stdout:
            if not "ADC" in line: print(line)
            if line.strip().endswith(".wav"):
                print(line)

            if line.strip().startswith("ADC"):
                tts_wav = line[4:].strip()
                break
        tts_time = time.time()
        print("HHHHHHHHH")
        voiceconv_alex_proc.stdin.write(tts_wav + "\t" + name + "\n")
        voiceconv_alex_proc.stdin.flush()
        print("Cloning ...")
        for line in voiceconv_alex_proc.stdout:
            if not "ADC:" in line:
                print(line)
            else:
                if line.strip().startswith("ADC"):
                    vc_wav = line[4:].strip()
                    break
        #   utt_id = str(uuid.uuid4())

        wav2lip_proc.stdin.write(vc_wav + "\t" + name + "\n")
        # Dogucan, please save the mp4 file with utt_id
        wav2lip_proc.stdin.flush()

        for line in wav2lip_proc.stdout:
            if not "RES" in line:
                print(line)
            else:
                video_file = line.split()[1]
                if video_file == "None":
                    video_file = ""
                break
        list_video_paths.append([video_file, speaker_idx, message])
        sLock.release()
    except:
        print("Something wrong")
    print("TTS time")
    print(tts_time - start)
    print("Total time")
    print(time.time() - start)
    return "Done"


@app.route('/list_videos', methods=['GET'])
def list_video():
    if len(list_video_paths) < 1: return "Empty"
    video_file = list_video_paths.pop(0)
    return video_file


@app.route('/tts', methods=['GET', 'POST'])
def predict():
    message = request.form.get('input_text')
    if message is None:
        return redirect((url_for("home")))
    message = message.strip() + "\n"
    name = request.form.get("input_name")
    video_name = request.form.get("input_video")
    start = time.time()
    try:
        tts_en_proc.stdin.write(message)
    except Exception as e:
        print(f"E: {e}")
    print("!1111!!")
    tts_en_proc.stdin.flush()
    print("ASD")
    tts_wav = ""
    for line in tts_en_proc.stdout:
        if not "ADC" in line: print(line)
        if line.strip().endswith(".wav"):
            print(line)

        if line.strip().startswith("ADC"):
            tts_wav = line[4:].strip()
            break
    tts_time = time.time()
    print("HHHHHHHHH")
    # print(tts_wav)
    voiceconv_alex_proc.stdin.write(tts_wav + "\t" + name + "\n")
    voiceconv_alex_proc.stdin.flush()
    print("Cloning ...")
    for line in voiceconv_alex_proc.stdout:
        if not "ADC:" in line:
            print(line)
        else:
            if line.strip().startswith("ADC"):
                vc_wav = line[4:].strip()
                break
    wav2lip_proc.stdin.write(vc_wav + "\t" + video_name + "\n")
    wav2lip_proc.stdin.flush()
    print("Wav2lip:::")
    for line in wav2lip_proc.stdout:
        if not "RES" in line:
            print(line)
        else:
            video_file = line.split()[1]
            break
    print("TTS time")
    print(tts_time - start)
    print("Total time")
    print(time.time() - start)
    # shutil.move("/project/OML/titanic/VoiceConv/converted_iwslt_4/converted_gen.wav","audio/converted_gen.wav")
    # get the description submitted on the web page
    return render_template('simple_template_2.html', voice="converted_gen.wav", TTS_voice="TTS_gen.wav",
                           video=video_file,
                           sample_text=message, model_choice=name, video_choice=video_name)
    # return 'Description entered: {}'.format(a_description)


@app.route('/<voice>', methods=['GET'])
def stream(voice):
    def generate():
        if os.path.exists(os.path.join('audio', voice)):
            with open(os.path.join('audio', voice), "rb") as fwav:
                data = fwav.read(1024)
                while data:
                    yield data
                    data = fwav.read(1024)

    if voice.endswith("wav"):
        return Response(generate(), mimetype="audio/")
    if voice.endswith("mp4"):
        return Response(generate(), mimetype="'video/mp4'")


def read_video():
    """
    Generate a random number every 2 seconds and emit to a socketio instance (broadcast)
    Ideally to be run in a separate thread?
    """
    while not thread_stop_event.isSet():
        if len(list_video_paths) < 1:
            socketio.emit('video_file_sender', {'video_file': "", 'video_speaker':"", 'video_transcript':""}, namespace='/test')
            socketio.sleep(1)
        else:
            video_info = list_video_paths.pop(0)
            video_file = video_info[0]
            video_speaker = video_info[1]
            video_transcript = video_info[2]
            print("sending video")
            socketio.emit('video_file_sender', {'video_file': video_file, 'video_speaker':video_speaker, 'video_transcript':video_transcript}, namespace='/test')


            cap = cv2.VideoCapture("audio/"+video_file)

            length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_dur = int(length/fps)+2
            print(video_dur)
            print(fps)
            print(length)

            socketio.sleep(video_dur)



@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    # Start the random number generator thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(read_video)


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
#    if request.method == 'POST':
#        prediction_data = request.json
#        print(prediction_data)
#    return jsonify({'result': prediction_data})

# boilerplate flask app code

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

if __name__ == "__main__":
    log_path = "./worker_log"
    tts_en_err_file = open(os.path.join(log_path, 'tts_en.log_'), 'a+')
    voiceconv_alex_err_file = open(os.path.join(log_path, 'voiceconv.log_'), 'a+')
    wav2lip_err_file = open(os.path.join(log_path, 'wav2lip.log_'), 'a+')

    tts_en_proc = subprocess.Popen('./process/inference_tts.sh {}'.format(4),
                                   shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=tts_en_err_file)
    for line in tts_en_proc.stdout:
        if "Up and running" in line:
            break

    voiceconv_alex_proc = subprocess.Popen('process/inference_vc.sh {}'.format(4),
                                           shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=voiceconv_alex_err_file)
    print("Starting Voice conv for Alex")

    for line in voiceconv_alex_proc.stdout:
        if "VoiceConv READY" in line:
            break

    wav2lip_proc = subprocess.Popen('process/inference_w2l.sh {}'.format(4),
                                    shell=True, encoding="utf-8", bufsize=0, universal_newlines=False,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=wav2lip_err_file)

    for line in wav2lip_proc.stdout:
        if "Wav2lip READY" in line:
            break

    # app.run(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
    socketio.run(app, debug=True, host="0.0.0.0", port=8080, use_reloader=False)
