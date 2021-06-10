import cv2
import os
import glob

path = os.path.abspath(os.getcwd())


def slice_video(input_video_path, write_frame_path):
    j = 0
    dir_path = os.path.join(write_frame_path, '{:05}'.format(j))
    while(os.path.isdir(dir_path) == True):
        j += 1
        dir_path = os.path.join(write_frame_path, '{:05}'.format(j))
    os.mkdir(dir_path)

    cap = cv2.VideoCapture(input_video_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        write_frame_path = os.path.join(dir_path, '{:05}'.format(i) + '.jpg')
        cv2.imwrite(write_frame_path, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def make_video(input_frame_path, write_video_path):
    img_array = []
    img_name_array = glob.glob(input_frame_path+'/*')
    img_name_array.sort()
    for filename in img_name_array:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    out = cv2.VideoWriter(write_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":
    input_video_path = os.path.join(path, 'test/input/55도발.mp4')
    output_frame_path = os.path.join(path, 'test/frame')
    input_frame_path = os.path.join(path, 'test/frame/00000')
    output_video_path = os.path.join(path, 'test/output/55도발_vsr.mp4')

    slice_video(input_video_path, output_frame_path)
    make_video(input_frame_path, output_video_path)
