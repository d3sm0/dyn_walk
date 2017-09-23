import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from memory.dataset import Dataset
from models.cvae import VAE
from models.fc_model import FCModel
from utils.misc_utils import load_data, train_test_set
from utils.tf_utils import _load, _save

tf.logging.set_verbosity(tf.logging.INFO)


class Imagination(object):
    def __init__(self, obs_dim, acts_dim, model_path=None, model='vae', z_dim=None):
        self.s, self.a = [], []
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.is_trained = False
        self.model_name = model
        if self.model_name == 'vae':
            assert z_dim is not None
            self.model = VAE(obs_dim=obs_dim, acts_dim=acts_dim, z_dim=z_dim, batch_size=64)
        elif self.model_name == 'fc':
            self.model = FCModel(obs_dim=obs_dim, acts_dim=acts_dim)

        self.model_path = model_path

    def set_state(self, state):
        self.state = state
        return self.state

    def step(self, a):
        s1_tilde = np.squeeze(self.model.step(self.state, a))

        reward, done = self.calc_reward(self.state, a, s1_tilde)
        self.state = s1_tilde

        return s1_tilde, reward, done, {}

    def collect(self, s, a):
        self.s.append(s)
        self.a.append(a)

    def train(self, iter_batch=20, dataset=None, verbose=False):
        losses = []
        i = 0
        if dataset is None:
            dataset = Dataset(data={
                'obs': np.vstack(self.s[:-1]),
                'acts': np.vstack(self.a[:-1]),
                'obs1': np.vstack(self.s[1:]),
            }, shuffle=True)

        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs']
                s1 = batch['obs1']
                acts = batch['acts']
                stats = self.model.train(s0, acts, s1)
                losses.append(stats)
                i += 1
                if verbose and i % 100 == 0:
                    tf.logging.info('Dataset covered {}. current loss {}'.format(i / dataset.n, stats))
        return {'model_avg_loss': np.mean(losses)}

    @staticmethod
    def calc_reward(s0, a, s1):
        # reward = 1.0
        # ob = s0
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone

        posbefore = s0[0]
        posafter, height, ang = s1[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        return reward, done

    def eval_model(self, dataset, iter_batch=1, verbose=False):
        losses = []
        i = 0
        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs']
                s1 = batch['obs1']
                acts = batch['acts']
                stats = self.model.eval(s0, acts, s1)
                i += 1
                losses.append(stats)
                if i % 100 == 0:
                    tf.logging.info('Dataset covered {}. current loss {}'.format(i / dataset.n, stats))
        return {'avg_loss': np.mean(losses)}

    def pretrain_model(self, data_dir, verbose=True):
        data = load_data(data_dir=data_dir)
        assert data is not None
        train_set, test_set = train_test_set(datasets=data)

        train_loss = self.train(
            dataset=Dataset(data={'obs': train_set[0], 'acts': train_set[1], 'obs1': train_set[2]}, shuffle=True),
            iter_batch=1, verbose=verbose)

        tf.logging.info('Testing results. Train loss: {}'.format(train_loss))
        test_loss = self.eval_model(
            dataset=Dataset(data={'obs': test_set[0], 'acts': test_set[1], 'obs1': test_set[2]}, shuffle=True),
            iter_batch=1, verbose=verbose)
        tf.logging.info('Pre train finished. Test loss: {}'.format(test_loss))

        _save(saver=self.model.saver, sess=self.model.sess, log_dir=self.model_path)
        self.is_trained = True

        return {'train_loss': train_loss, 'test_loss': test_loss}

    def load(self):
        print("loading model")
        _load(saver=self.model.saver, sess=self.model.sess, log_dir=self.model_path)
        print("loaded model")


if __name__ == "__main__":

    from utils.running_stats import ZFilter
    from worker import Worker
    import json

    with open('config.json') as f:
        config = json.load(f)

    dataset_path = 'log-files/InvertedPendulum-v1/Sep-22_15_10'  # 'log-files/InvertedPendulum-v1/Sep-22_13_29'
    worker = Worker(config, log_dir=dataset_path)
    # _load(sess = worker.agent.sess, saver = worker.agent.saver,log_dir='log-files/old_logs/Sep-19_01_19')
    img = Imagination(obs_dim=worker.env_dim[0], acts_dim=worker.env_dim[1], model_path='tf-models/', model='vae',
                      z_dim=2)

    # dataset = Imagination.build_dataset(log_dir=dataset_path + '/dataset')

    try:
        img.load()
    except Exception as e:
        tf.logging.info('Pre training model')
        img.pretrain_model(data_dir=dataset_path + '/dataset')

    ob_filter = ZFilter((worker.env_dim[0],))

    TEST_RUNS = 1000
    worker.warmup(ob_filter, 60)
    # img.reset(ob1)
    losses_v = []
    obs = []
    losses_v = 0
    losses_r = 0
    obs = []
    ep_loss_v = []
    ep_loss_r = []
    for _ in range(TEST_RUNS):
        done = False
        ob = worker.env.reset()
        img.set_state(ob)
        while not done:
            # worker.env.render()
            ob = ob_filter(ob)

            act, _ = worker.agent.get_action_value(ob)

            # img.set_state(ob)  # follow the ground truth
            ob1_tilde, reward_tilde, done, _ = img.step(act)

            ob1, r, done, _ = worker.env.step(act)
            obs.append((ob1_tilde[1] - ob1[1]))
            v = worker.agent.get_value([ob1])
            ob1 = 0
            loss_r = 0.5 * np.square(reward_tilde - r)

            v_tilde = worker.agent.get_value([ob1_tilde])

            loss_v = 0.5 * np.square(v_tilde - v)
            losses_v += loss_v
            losses_r += loss_r

            ob = ob1_tilde.copy()
        ep_loss_v.append(losses_v)
        ep_loss_r.append(losses_r)
        losses_r = losses_v = 0

    tf.logging.info('Eval ended {}, {}'.format(np.mean(ep_loss_r), np.mean(ep_loss_v)))
    plt.plot(ep_loss_v)
    plt.savefig("done")
    plt.close()
    plt.plot(ep_loss_v)
    plt.savefig("ep_loss-v")
    plt.close()
    plt.plot(ep_loss_r)
    plt.savefig("ep_loss_r")
