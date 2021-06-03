import time

import tensorflow as tf

from dataset import load_data
from nets import OurModel

BATCH_SIZE = 32

checkpoint_path = "checkpoint/FR_16_4." + str(int(time.time())) + ".{epoch:03d}-{val_loss:.5f}"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.0001,
    first_decay_steps=40
)


def train_and_evaluate():
    print("[+] init lr 0.0001 / decay step 40 / huber 1.35 / train ratio 0.9")
    strategy = tf.distribute.MirroredStrategy()
    (x_train, y_train), (x_valid, y_valid) = load_data(train_ratio=0.9)

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
            loss=tf.keras.losses.Huber(delta=1.35)
        )

        history = model.fit(
            train_data,
            batch_size=BATCH_SIZE,
            epochs=128,
            validation_data=val_data,
            callbacks=[checkpoint_callback]
        )
        model.save(f"./FR_16_4_{str(int(time.time()))}", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()
