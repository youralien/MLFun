# Revamp of Experiment saver. Original from
# https://github.com/lukemetz/cuboid/blob/master/cuboid/extensions/__init__.py
import shutil
import os
from blocks.dump import save_parameter_values
import datetime

from blocks.extensions import SimpleExtension, TrainingExtension
import json

class ExperimentSaver(SimpleExtension):
    """
    Save a given experiment to a file
    * dump the current source directory
    * save parameters
    * dump the training logs

    Parameters
    ---------
    dest_directory: basestring
        Path to dump the experiment.
    src_directory: basestring
        Path to source to be copied.
    """

    def __init__(self, dest_directory, src_directory, **kwargs):
        self.dest_directory = dest_directory
        self.src_directory = src_directory

        self.params_path = os.path.join(self.dest_directory, 'params')
        self.log_path = os.path.join(self.dest_directory, 'log.csv')

        self.write_src()

        kwargs.setdefault("after_epoch", True)

        super(ExperimentSaver, self).__init__(**kwargs)

    def params_path_for_epoch(self, i):
        return os.path.join(self.params_path, str(i))

    def write_src(self):
        # Don't overwrite anything, move it to a backup folder
        if os.path.exists(self.dest_directory):
            time_string = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
            move_to = self.dest_directory + time_string + "_backup"
            shutil.move(self.dest_directory, move_to)

        os.mkdir(self.dest_directory)
        src_path = os.path.join(self.dest_directory, 'src')

        os.mkdir(self.params_path)

        def ignore(path, names):
            # TODO actually manage paths correctly
            if path == self.dest_directory or path == './' + self.dest_directory:
                return names
            else:
                return []

        shutil.copytree(self.src_directory, src_path, ignore=ignore)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        epoch_done = log.status['epochs_done']
        params = self.main_loop.model.get_param_values()
        path = self.params_path_for_epoch(epoch_done)
        save_parameter_values(params, path)

        log.to_dataframe().to_csv(self.log_path)
