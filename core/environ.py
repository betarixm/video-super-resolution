import time
import tensorflow as tf

EPOCHS = 128
BATCH_SIZE = 16
INIT_LR = 0.01
FIRST_DECAY_STEPS = 50
TRAIN_DATASET_RATIO = 0.9
HUBER_DELTA = 200.0
ID = str(int(time.time()))

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=INIT_LR,
    first_decay_steps=FIRST_DECAY_STEPS
)