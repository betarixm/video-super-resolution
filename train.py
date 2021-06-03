import time

import tensorflow as tf

from dataset import load_data
from nets import OurModel

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
    (X_train, y_train), (X_valid, y_valid) = load_data(train_ratio=0.9)
    with strategy.scope():
        model = OurModel()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.Huber(delta=1.35)
        )

        history = model.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=128,
            validation_data=(X_valid, y_valid),
            callbacks=[checkpoint_callback]
        )
        model.save(f"./FR_16_4_{str(int(time.time()))}", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()
