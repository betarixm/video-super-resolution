import pickle
import glob
import multiprocessing
import random

import numpy as np

from worker import worker, m_dir_hr_files, m_dir_lr_files, NUM_DIR

DIR_NUM = 300
pool = multiprocessing.Pool()


def load_data(train_ratio=0.75, num_dir=None, checkpoint=False, is_save=False):
    if checkpoint:
        print("[+] Loading (checkpoint)...", end=" ", flush=True)
        with open('./dataset.LR.pickle', 'rb') as fr:
            dir_lr_files = pickle.load(fr)
        with open('./dataset.HR.pickle', 'rb') as fr:
            dir_hr_files = pickle.load(fr)
        print(dir_lr_files)
        print(dir_hr_files)
        print("Done!", flush=True)
    else:
        print("[+] Dataset Crawling...", end=' ', flush=True)

        if num_dir is None:
            dir_names_x, dir_names_y = glob.glob('./input/LR/*'), glob.glob('./input/HR/*')
        else:
            dir_names_x, dir_names_y = glob.glob(f'./input/LR/{num_dir}'), glob.glob(f"./input/HR/{num_dir}")

        dir_names_x.sort(), dir_names_y.sort()
        dir_names_x = random.choices(dir_names_x, k=DIR_NUM)
        dir_names_y = random.choices(dir_names_y, k=DIR_NUM)
        dir_inputs_x = [glob.glob(f"{d}/*") for d in dir_names_x]
        dir_inputs_y = [glob.glob(f"{d}/*") for d in dir_names_y]

        print(f"[+] Selected sets: {', '.join([x.split('/')[-1] for x in dir_names_x])}")

        target_x = [str(file) for d in dir_inputs_x for file in d]
        target_y = [str(file) for d in dir_inputs_y for file in d]

        pool.map(worker, target_x + target_y)
        pool.close()
        pool.join()
        print("Done!", flush=True)

        dir_lr_files = []
        dir_hr_files = []

        for fl in m_dir_lr_files:
            dir_lr_files.append(dict(fl))
            del fl

        for fl in m_dir_hr_files:
            dir_hr_files.append(dict(fl))
            del fl

        if is_save:
            print("[+] Saving (checkpoint)...", end=' ', flush=True)
            with open('dataset.LR.pickle', 'wb') as fr:
                pickle.dump(dir_lr_files, fr)
            with open('dataset.HR.pickle', 'wb') as fr:
                pickle.dump(dir_hr_files, fr)
            print("Done!", flush=True)

    m_dir_files_x, m_dir_files_y = [[] for _ in range(NUM_DIR)], [[] for _ in range(NUM_DIR)]

    for d_key, d_dict in enumerate(dir_lr_files):
        for i in range(len(d_dict)):
            m_dir_files_x[d_key].append(d_dict[i])

    for d_key, d_dict in enumerate(dir_hr_files):
        for i in range(len(d_dict)):
            m_dir_files_y[d_key].append(d_dict[i])

    x_train, y_train = [], []
    for d_x, d_y in zip(m_dir_files_x, m_dir_files_y):
        try:
            assert len(d_x) == len(d_y)
        except AssertionError as e:
            print(f"[E] len(d_x): {len(d_x)} len(d_y): {len(d_y)}")

        for i in range(len(d_x) - 6):
            x_train.append(d_x[i:i + 7])
            y_train.append(d_y[i + 3])

    print(f"[+] X: {len(m_dir_files_x)}, {len(x_train)}, Y: {len(m_dir_files_y)}, {len(y_train)}")

    x_train, y_train = np.asarray(x_train), np.asarray(y_train)

    print(f"[+] {len(x_train)}")
    x_valid = x_train[int(len(x_train) * train_ratio):]
    y_valid = y_train[int(len(y_train) * train_ratio):]
    x_train = x_train[:int(len(x_train) * train_ratio)]
    y_train = y_train[:int(len(y_train) * train_ratio)]

    print("Done!", flush=True)
    return (x_train, y_train), (x_valid, y_valid)


if __name__ == "__main__":
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(load_data(checkpoint=True, is_save=True), f)