from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import scipy.io.wavfile as swav
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time
import base64

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--name', type=str,
                    help='Speaker indentity of facepath', required=True)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outpath', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                         'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--read_from_folder', default=False, action='store_true',
                    help='Read from file without running face detection')

parser.add_argument('--resize_all', default=False, action='store_true', help='resize all frames after extracted')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


class Face:
    def __init__(self, name, face_path):
        self.name = name
        self.face_path = face_path
        self.initialize()
        self.end_frame_previous = 0
        self.start_directly_from_reverse = False

    def initialize(self):
        if not os.path.isfile(self.face_path):
            # raise ValueError('--face argument must be a valid path to video/image file')
            return
        if self.face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.full_frames = [cv2.imread(self.face_path)]
            self.fps = args.fps

        else:
            video_stream = cv2.VideoCapture(self.face_path)
            self.fps = video_stream.get(cv2.CAP_PROP_FPS)
            print("fps={}".format(self.fps))

            print('Reading video frames...')

            self.full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if args.resize_factor > 1:
                    frame = cv2.resize(frame,
                                       (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

                if args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                self.full_frames.append(frame)
            """
            h, w, _ = self.full_frames[0].shape
            if h > 480 and w > 480 and args.resize_all:
                if h < w:
                    min_size = h
                    max_size = w
                else:
                    min_size = w
                    max_size = h
                coefficient = float(min_size) / float(480)
                new_h = int(h/coefficient)
                new_w = int(w/coefficient)
                self.full_frames = [cv2.resize(x, (w, h)) for x in self.full_frames]
            """

    def detect_face(self):
        if args.read_from_folder:
            f = open(os.path.join(self.face_path.split(".mp4")[0], "coords.txt"), "r")
            all_data_from_file = f.readlines()
            f.close()
            self.resize_needed = False
            if "," not in all_data_from_file[0]:
                print("it is in the resize area to get new sizes")
                self.resize_needed = True
                new_size = all_data_from_file[0].rstrip().split(" ")
                self.new_w = int(new_size[0])
                self.new_h = int(new_size[1])
                all_data_from_file.pop(0)
                self.full_frames = [cv2.resize(x, (self.new_w, self.new_h)) for x in self.full_frames]

            image_names_from_file = [self.face_path.split(".mp4")[0] + "/" + x.split(",")[0] + ".png" for x in
                                     all_data_from_file]
            y1s = [int(x.split(",")[-1].split(" ")[0]) for x in all_data_from_file]
            y2s = [int(x.split(",")[-1].split(" ")[1]) for x in all_data_from_file]
            x1s = [int(x.split(",")[-1].split(" ")[2]) for x in all_data_from_file]
            x2s = [int(x.split(",")[-1].split(" ")[3]) for x in all_data_from_file]
            self.face_det_results = []

            for i in range(len(image_names_from_file)):
                imgs = cv2.imread(image_names_from_file[i])
                self.face_det_results.append([imgs, (y1s[i], y2s[i], x1s[i], x2s[i])])
        else:
            if args.box[0] == -1:
                if not args.static:
                    self.face_det_results = face_detect(self.full_frames)  # BGR2RGB for CNN face detection
                else:
                    self.face_det_results = face_detect([self.full_frames[0]])
            else:
                print('Using the specified bounding box instead of face detection...')
                y1, y2, x1, x2 = args.box
                self.face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in self.full_frames]


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    avg_detection_time = 0.0
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                start_face_time = time.time()
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
                end_face_time = time.time()
                avg_detection_time += (end_face_time - start_face_time)
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    print("Avg face detection time: {}".format(avg_detection_time / len(images)))

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels, face_det_results, facebox_object=None):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    start_idx_faces = facebox_object.end_frame_previous

    if facebox_object.start_directly_from_reverse:
        frames = frames[::-1]
        face_det_results = face_det_results[::-1]

    for i, m in enumerate(mels):
        if i < len(frames) - start_idx_faces:
            idx = i % len(frames) + start_idx_faces
        else:
            idx = (i + start_idx_faces) % len(frames)
            if idx == 0:
                frames = frames[::-1]
                face_det_results = face_det_results[::-1]
                facebox_object.start_directly_from_reverse = not facebox_object.start_directly_from_reverse

        # idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def main():
    # if not args.audio.endswith('.wav'):
    #     print('Extracting raw audio...')
    #     command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
    #
    #     subprocess.call(command, shell=True)
    #     args.audio = 'temp/temp.wav'
    list_facepath = args.face.split("|")
    list_name = args.name.split("|")
    list_facebox = []
    for name, facepath in zip(list_name, list_facepath):
        facebox = Face(name, facepath)
        facebox.detect_face()
        list_facebox.append(facebox)

    w2l_model = load_model(args.checkpoint_path)
    print("Model loaded")

    print("Wav2lip READY")
    i = 0
    for line in sys.stdin:
        # print("start to process stdin request")
        # print(line)
        start_of_the_lip_part = time.time()
        i += 1
        # src_wav_path = pair["source"]
        if not line:
            continue
        adc, name = line.split("\t")
        # print("name: {}" .format(name))
        if name.strip() in list_name:
            facebox = list_facebox[list_name.index(name.strip())]
        else:
            print("Can not find name")
            facebox = list_facebox[0]
        # print("Number of frames available for inference: " + str(len(facebox.full_frames)))
        # print(name.strip())
        face_det_results = facebox.face_det_results
        fps = facebox.fps
        full_frames = facebox.full_frames
        wav = np.frombuffer(base64.b64decode(adc.strip()), dtype="float32")
        # wav = (torch.from_numpy(wav_)).type(torch.FloatTensor)
        # print(f"STDIN: {wav.shape}")
        # print(f"STDIN: {wav.min()},{wav.max()},{wav.mean()}")
        #  wav = audio.load_wav(args.audio, 16000)

        # samplerate, data = swav.read(args.audio)
        # swav.write('waibel_german_13s_21s.wav',samplerate,data)

        mel = audio.melspectrogram(wav)
        # print(mel.shape)
        if mel.shape[1] < mel_step_size: mel = np.concatenate([mel, np.zeros((80, mel_step_size - mel   .shape[1]))],
                                                              axis=1)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        # print("Length of mel chunks: {}".format(len(mel_chunks)))

        # full_frames_in = full_frames[:len(mel_chunks)]
        # full_frames_in_ = full_frames[:len(mel_chunks)]

        # face_det_results_in_ = face_det_results[:len(mel_chunks)]

        # first approach
        """
        full_frames_in_ = full_frames[facebox.end_frame_previous+1:] + full_frames[:facebox.end_frame_previous+1]
        full_frames_in = full_frames_in_[:len(mel_chunks)]
        face_det_results_in_ = face_det_results[facebox.end_frame_previous+1:] + face_det_results[:facebox.end_frame_previous+1]
        face_det_results_in = face_det_results_in_[:len(mel_chunks)]
        """

        # second approach
        full_frames_in = full_frames.copy()
        face_det_results_in = face_det_results.copy()

        batch_size = args.wav2lip_batch_size

        avg_time = 0.0

        total_iter = int(np.ceil(float(len(mel_chunks)) / batch_size))

        gen = datagen(full_frames_in.copy(), mel_chunks, face_det_results_in, facebox)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(
                                                                            np.ceil(
                                                                                float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                if facebox.resize_needed:
                    frame_h = int(facebox.new_h)
                    frame_w = int(facebox.new_w)
                    # print("frame size: {}, {}" .format(frame_h, frame_w))
                else:
                    frame_h, frame_w = full_frames_in[0].shape[:-1]
                write_name_video_file = name.strip() + '_' + str(facebox.end_frame_previous + 1) + '_' + str(
                    int(total_iter * batch_size) + facebox.end_frame_previous) + '.avi'
                out = cv2.VideoWriter('temp/' + write_name_video_file,
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
                # start_write_name = str(facebox.end_frame_previous + 1)
                facebox.end_frame_previous = int(total_iter * batch_size) + facebox.end_frame_previous

            # print("img batch: {}, mel batch: {}".format(img_batch.shape, mel_batch.shape))
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                start_time = time.time()
                pred = w2l_model(mel_batch, img_batch)
                end_time = time.time()
                # print("time consumption:")
                # print(end_time-start_time)
                avg_time += (end_time - start_time)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                # print("analysis of frame details. y1: {}, y2:{}, x1:{}, x2:{}, frame shape:{}, new_h:{}, new_w:{}" .format(y1, y2, x1, x2, f.shape, frame_h, frame_w))
                out.write(f)

        out.release()
        print("avg running time per frame: {}".format(avg_time / len(mel_chunks)))

        os.makedirs("temp", exist_ok=True)
        # command = 'ffmpeg -y -i {} -i {} -strict -2  -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
        # command = 'ffmpeg -y -i {} -i {} -strict -2  -q:v 1 {}'.format(args.audio, 'temp/'  + write_name_video_file, args.outfile)
        command = 'ffmpeg -y -i {} -i {} -strict -2  -q:v 1 {}'.format(args.audio, 'temp/' + write_name_video_file,
                                                                       "/home/namnguyen/PycharmProjects/my_first_flask_app/audio/" +
                                                                       write_name_video_file.split(".avi")[0] + ".mp4")
        # start = time.time()
        os.system(command)
        # print("time for ffmpeg")
        # print(time.time() - start)
        # sys.stdout.write("RES: \n")
        end_of_the_lip_part = time.time()
        print_time = end_of_the_lip_part - start_of_the_lip_part
        print("Total full part of the lip generation: {}".format(print_time))
        return_text = "RES: " + write_name_video_file.split(".avi")[0] + ".mp4\n"
        sys.stdout.write(return_text)
        sys.stdout.flush()
        # print("after stdout")
        # sys.stdout.write("RES: \n")
        # sys.stdout.flush()


if __name__ == '__main__':
    main()
