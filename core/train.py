import time

import tensorflow as tf

from core.dataset import load_data
from core.nets import OurModel

EPOCHS = 128
BATCH_SIZE = 16
INIT_LR = 0.01
FIRST_DECAY_STEPS = 50
TRAIN_DATASET_RATIO = 0.9
HUBER_DELTA = 200.0
ID = str(int(time.time()))

checkpoint_path = "checkpoint/FR_16_4." + ID + ".{epoch:03d}-{val_loss:.5f}"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=INIT_LR,
    first_decay_steps=FIRST_DECAY_STEPS
)


def train_and_evaluate():
    print(f"[+] Training Information\n"
          f"    ID:               {ID}\n"
          f"    Epochs:           {EPOCHS}\n"
          f"    Batch size:       {BATCH_SIZE}\n"
          f"    Initial LR:       {INIT_LR}\n"
          f"    First decay step: {FIRST_DECAY_STEPS}\n"
          f"    Training set:     {TRAIN_DATASET_RATIO}\n"
          f"    Huber delta:      {HUBER_DELTA}\n")

    strategy = tf.distribute.MirroredStrategy()
    (x_train, y_train), (x_valid, y_valid) = load_data(train_ratio=TRAIN_DATASET_RATIO)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    train_data, val_data = train_data.batch(BATCH_SIZE), val_data.batch(BATCH_SIZE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_data, val_data = train_data.with_options(options), val_data.with_options(options)

    with strategy.scope():
        model = OurModel()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.Huber(delta=HUBER_DELTA)
        )

        history = model.fit(
            train_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_data,
            callbacks=[checkpoint_callback]
        )
        model.save(f"./FR_16_4_{ID}", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()
