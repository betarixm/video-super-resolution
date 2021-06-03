import tensorflow as tf

from dataset import load_data
from nets import OurModel

checkpoint_path = "checkpoint/FR_16_4.{epoch:03d}-{val_loss:.2f}"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=100
)


def train_and_evaluate():
    strategy = tf.distribute.MirroredStrategy()
    (X_train, y_train), (X_valid, y_valid) = load_data()
    with strategy.scope():
        model = OurModel()
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule),
            loss=tf.keras.losses.Huber(delta=0.01)
        )

        history = model.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=128,
            validation_data=(X_valid, y_valid),
            callbacks=[checkpoint_callback]
        )
        model.save("./FR_16_4", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()



