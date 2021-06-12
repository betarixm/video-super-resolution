import time
import os
import glob
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from uwsgidecorators import thread

from app.environ import UPLOAD_FOLDER, PROCESSED_FOLDER, PROCESSING_FOLDER
from core.nets import OurModel
from core.environ import lr_schedule, HUBER_DELTA

running = {}

weight_path = os.path.join('checkpoint', 'FR_16_4.1622873040.034-0.00526')


def split_video(session_id: str, filename: str):
    """
    Original video file is in "uploads/{session_id}/{filename}"
    Read the video file and divide it into frames and blocks, again.

    Save these to "processing/{session_id}/{filename}/{block_number}/{frame_number}/000.png
    -> changed to "processing/{session_id}/{filename}/{row_number}/{column_number}/00000.png
    Note that our deep learning model uses frame-major order dataset, so those would saved as that convention.

    Warning: if the total number of frame exceeds 10,000 then the order of frames can be strange!!
    """

    input_video_path = os.path.join(UPLOAD_FOLDER, str(session_id), filename)
    write_frame_dir_path = os.path.join(PROCESSING_FOLDER, str(session_id), filename)
    if not os.path.exists(PROCESSING_FOLDER):
        os.mkdir(PROCESSING_FOLDER)

    if not os.path.exists(f"{PROCESSING_FOLDER}/{session_id}"):
        os.mkdir(f"{PROCESSING_FOLDER}/{session_id}")

    if not os.path.exists(write_frame_dir_path):
        os.mkdir(write_frame_dir_path)

    cap = cv2.VideoCapture(input_video_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        n_h = int(height / 32) if height % 32 != 0 else int(height / 32) - 1
        n_w = int(width / 32) if width % 32 != 0 else int(width / 32) - 1
        h_pixel_to_pad = height - n_h * 32 if height % 32 != 0 else 0
        w_pixel_to_pad = width - n_w * 32 if width % 32 != 0 else 0
        frame = cv2.copyMakeBorder(frame, 0, h_pixel_to_pad, 0, w_pixel_to_pad, cv2.BORDER_REPLICATE)
        write_frame_path = os.path.join(write_frame_dir_path, '{:05}'.format(i))
        if not os.path.isdir(write_frame_path):
            os.mkdir(write_frame_path)
        for j in range(n_h + 1):
            write_frame_row_path = os.path.join(write_frame_path, '{:05}'.format(j))
            if not os.path.isdir(write_frame_row_path):
                os.mkdir(write_frame_row_path)
            for k in range(n_w + 1):
                write_frame_column_path = os.path.join(write_frame_row_path, '{:05}'.format(k) + '.jpg')
                img = frame[32 * j:32 * (j + 1), 32 * k:32 * (k + 1)]
                cv2.imwrite(write_frame_column_path, img)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


@thread
def process(session_id: str, filename: str):
    if session_id not in running:
        running[session_id] = []

    if filename in running[session_id]:
        return False

    running[session_id].append(filename)
    """
    Using result of split_video, run tensorflow model.
    Merge and save these result into "processed/{session_id}/{filename}"
    """
    split_video(session_id, filename)

    # Load Frames into img_array
    input_frame_path = os.path.join(PROCESSING_FOLDER, str(session_id), filename)
    img_array = []
    input_frame_path_array = glob.glob(input_frame_path + '/*')
    input_frame_path_array.sort()
    for frame_path in input_frame_path_array:
        num_frame = len(input_frame_path_array)
        input_row_path_array = glob.glob(frame_path + '/*')
        input_row_path_array.sort()
        for row_path in input_row_path_array:
            num_row = len(input_row_path_array)
            input_column_path_array = glob.glob(row_path + '/*')
            input_column_path_array.sort()
            for file in input_column_path_array:
                num_column = len(input_column_path_array)
                img = cv2.imread(file)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)

    input_img_array = []
    for i in range(num_frame - 6):
        for j in range(num_row):
            for k in range(num_column):
                frame_bunch = []
                for l in range(i, i + 7):
                    frame_bunch.append(img_array[l * num_row * num_column + j * num_column + k])
                input_img_array.append(frame_bunch)
    input_img_array = np.asarray(input_img_array, dtype=np.float32)

    # Run Model
    # model = tf.keras.models.load_model(
    #     weight_path,
    #     custom_objects={
    #         "FR_16": FR_16,
    #         "DynFilter": DynFilter,
    #         "depth_to_space_3D": depth_to_space_3D
    #     }
    # )
    model = OurModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA)
    )
    model.load_weights(weight_path)
    img_array_output = model.predict(input_img_array)
    # Merge into video
    merged_img_array = []
    for i in range(num_frame - 6):
        merged_image = Image.new('RGB', (num_column * 128, num_row * 128))
        for j in range(num_row):
            for k in range(num_column):
                img = Image.fromarray(
                    img_array_output[i * num_row * num_column + j * num_column + k][0].astype('uint8'))
                merged_image.paste(img, (k * 128, j * 128))
        merged_img_array.append(merged_image)

    write_video_path = os.path.join(PROCESSED_FOLDER, str(session_id), filename)

    if not os.path.exists(PROCESSED_FOLDER):
        os.mkdir(PROCESSED_FOLDER)

    if not os.path.exists(f"{PROCESSED_FOLDER}/{session_id}"):
        os.mkdir(f"{PROCESSED_FOLDER}/{session_id}")

    out = cv2.VideoWriter(write_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (num_column * 128, num_row * 128))
    for i in range(len(merged_img_array)):
        out.write(np.array(merged_img_array[i]))
    out.release()
    running[session_id].remove(filename)
