from tensorflow.keras.callbacks import Callback


class LearningRateReduce(Callback):

    def __init__(self, rate=0.99):
        super().__init__()
        self.rate = rate

    def on_epoch_end(self, epoch, logs=None):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * self.rate
        print("\nReducing learning rate from {} to {}".format(old_lr, new_lr))
        self.model.optimizer.lr = new_lr
