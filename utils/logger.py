import os
import csv
import json
from datetime import datetime

class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, env_name):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        now = datetime.utcnow().strftime("%b-%d_%H_%M")  # create unique dir

        self.main_path = os.path.join('log-files', env_name, now)
        os.makedirs(self.main_path)
        # filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        # for filename in filenames:     # for reference
        #     shutil.copy(filename, path)
        path = os.path.join(self.main_path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def save_experiment(self, config):
        try:
            with open(os.path.join(self.main_path, 'readme.txt'), 'w') as f:
                json.dump(config, f)
        except OSError or IOError:
            raise

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""

        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {} *****'.format(log['total_ep']))
        for key in log_keys:
            # if key[0] != '_':  # don't display log items with leading '_'
            print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
