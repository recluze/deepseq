import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from utils import get_current_timestamp
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

import logging

def visualize_history(hist, filename, plot_auc = False):
    fig, axs = plt.subplots(figsize=(15,15))
    plt.axis('off')

    plots = [('Losses', hist['loss'], hist['val_loss']),
             ('Accuracy', hist['acc'], hist['val_acc']),
             ('Recall', hist['recall'], hist['val_recall']),
             ('Precision', hist['precision'], hist['val_precision']),
             ('F1 Score', hist['fbeta_score'], hist['val_fbeta_score'])]

    if plot_auc:
        plots.append(('AUC', hist['auc'], hist['auc']))

    for n, i in enumerate(plots):
        ax1 = fig.add_subplot(320 + n + 1)
        plt.ylabel(i[0])
        # plt.ylim((0., 1.))
        x = [item + 1 for item in range(len(i[1]))] # correct xlabels
        ax1.plot(x, i[1], label="Train")
        ax1.plot(x, i[2], label="Test")

        # move legend below the plots
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    plt.savefig(filename)
    plt.close()


class LossHistory(Callback):
    def __init__(self, filename, model_snapshot_directory, exp_id):
        self.filename = filename
        self.history = {}
        self.best_fbeta_score = 0.0
        self.model_snapshot_directory = model_snapshot_directory
        self.exp_id = exp_id

    def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))
        # print logs.keys()
        pass

    def on_epoch_end(self, epoch, logs={}):
        # output
        metrics_line = ''
        for s in ['loss', 'acc', 'precision', 'recall', 'fbeta_score']:
            metrics_line += "%s: %.5f %s: %.5f - " %(s, logs[s], 'val_'+s, logs['val_' +s])

        logging.info(metrics_line)

        # save model if best
        new_fbeta_score = logs['val_fbeta_score']
        if new_fbeta_score > self.best_fbeta_score:
            self.best_fbeta_score = new_fbeta_score
            # also save the best model
            model_save_filename = self.model_snapshot_directory + '/'+ self.exp_id + \
                                        '-saved-model.h5'
            self.model.save(model_save_filename)
            logging.info("-- saved best model with f measure: %.5f on epoch: %d" % (self.best_fbeta_score, epoch))


        # update history with new information
        for k in logs.keys():
            try:
                self.history[k].append(logs[k])
            except KeyError:
                self.history[k] = [logs[k]]

        # print self.history
        # update file
        visualize_history(self.history, self.filename)
