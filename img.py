import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.spatial import KDTree

from memory.dataset import Dataset
from models.cvae import VAE
from models.fc_model import FCModel
from utils.misc_utils import load_data, train_test_set
from utils.tf_utils import _load, _save

tf.logging.set_verbosity(tf.logging.INFO)


class Imagination(object):
    def __init__(self, obs_dim, acts_dim, model_path=None, model='vae', z_dim=None, is_recurrent=False):
        self.s, self.a = [], []
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.is_trained = False
        self.model_name = model
        if self.model_name == 'vae':
            assert z_dim is not None
            self.model = VAE(obs_dim=obs_dim, acts_dim=acts_dim, z_dim=z_dim, batch_size=64, is_recurrent=is_recurrent)
        elif self.model_name == 'fc':
            self.model = FCModel(obs_dim=obs_dim, acts_dim=acts_dim, is_recurrent=is_recurrent)

        self.model_path = model_path

        import os
        os.makedirs(self.model_path, exist_ok=True)

    def set_state(self, state):
        self.state = state
        return self.state

    def step(self, a):
        s1_tilde = np.squeeze(self.model.step(self.state, a))

        reward, done = self.calc_reward(self.state, a, s1_tilde)
        self.state = s1_tilde
        done = False

        return s1_tilde, reward, done, {}

    def collect(self, s, a):
        self.s.append(s)
        self.a.append(a)

    def train(self, iter_batch=20, dataset=None, verbose=False, shuffle=True):
        l1s, l2s = [], []
        i = 0
        if dataset is None:
            dataset = Dataset(data={
                'obs': np.vstack(self.s[:-1]),
                'acts': np.vstack(self.a[:-1]),
                'obs1': np.vstack(self.s[1:]),
            }, shuffle=shuffle)

        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs']
                s1 = batch['obs1']
                acts = batch['acts']
                l1 = self.model.train(s0, acts, s1, m=0)
                i += 1
                l1s.append(l1)
                if verbose and i % 100 == 0:
                    tf.logging.info('Dataset covered {}. current loss {}'.format(i / dataset.n, np.mean(l1s)))




        # Xs = np.concatenate((dataset.data['obs'], dataset.data['acts']), axis=1)
        # # double check this
        # self.tree = KDTree(Xs)
        # l2s = [0, 0]
        i = 0
        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs']
                s1 = batch['obs1']
                acts = batch['acts']
                # where m = 1 states the last layer activation
                l2 = self.model.train(s0, acts, s1, m=1)
                l2s.append(l2)
                i += 1
                if verbose and i % 100 == 0:
                    tf.logging.info('Dataset covered {}. current loss {}'.format(i / dataset.n, np.mean(l2s)))


        return {'model_avg_loss': np.array(l1s).mean(), 'conf_avg_loss': np.array(l2s).mean()}

    @staticmethod
    def calc_reward(s0, a, s1):
        reward = 1.0
        ob = s0
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone

        # posbefore = s0[0]
        # posafter, height, ang = s1[0:3]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore)
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
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

    def pretrain_model(self, data_dir, concat_frames=1, verbose=True, shuffle=True):
        data = load_data(data_dir=data_dir)
        assert data is not None
        train_set, test_set = train_test_set(datasets=data, concat_frames=concat_frames)

        train_loss = self.train(
            dataset=Dataset(data={'obs': train_set[0], 'acts': train_set[1], 'obs1': train_set[2]}, shuffle=shuffle),
            iter_batch=1, verbose=verbose)
        tf.logging.info('Testing results. Train loss: {}'.format(train_loss))

        test_loss = self.eval_model(
            dataset=Dataset(data={'obs': test_set[0], 'acts': test_set[1], 'obs1': test_set[2]}, shuffle=shuffle),
            iter_batch=1, verbose=verbose)
        tf.logging.info('Pre train finished. Test loss: {}'.format(test_loss))

        _save(saver=self.model.saver, sess=self.model.sess, log_dir=self.model_path)
        self.is_trained = True
        return {'train_loss': train_loss, 'test_loss': test_loss}

    def load(self, data_dir=None):
        try:
            _load(saver=self.model.saver, sess=self.model.sess, log_dir=self.model_path)
        except Exception as e:
            tf.logging.error('Failed model to restore, retraining')
            if data_dir is not None:
                self.pretrain_model(data_dir)
            else:
                self.is_trained = False
                tf.logging.error('Dataset not found. The model will not be used in this run')

    def confidence(self, obs, acts):
        # conf_tilde = self.sess.run(self.conf_tilde,
        #                            feed_dict={self.obs: [obs],
        #                                       self.acts: [acts]})
        # return 1-abs(conf_tilde)
        x = np.concatenate((obs, acts),axis = 0)
        distance, idx = self.tree.query(x)
        # nn = self.tree.query(x  )
        # d = spatial.distance.euclidean(nn, x)
        return 1-np.clip(distance, 0,1)


    def train_conf_kdtree(self, obs, acts, obs1):
        self.tree = KDTree(np.concatenate((obs, acts), axis=1))


if __name__ == "__main__":

    import json
    with open('config.json') as f:
        config = json.load(f)

    dataset_path = 'log-files/InvertedPendulum-v1/Sep-24_01_41LOG_BRANCH_WIDTH::4LOG_BRANCH_DEPTH::1'
    data_dir = dataset_path + '/dataset'
    concat_frames = config['CONCATENATE_FRAMES']
    import pickle
    with open(data_dir + '/dataset.pkl', "rb") as fin:
        datasets = pickle.load(fin)
    data = datasets


    obs = data['obs']
    acts = data['acts']
    Xs = np.concatenate((obs, acts), axis = 1)
    idxs = np.random.randint(0,Xs.shape[0], size = 2)
    sx = Xs[idxs]
    print(np.linalg.norm(sx, axis =0))


    # tree = KDTree(Xs)
    # import random
    #
    # for x in Xs:
    #
    #     x = x + np.random.normal(size=x.shape[0])
    #     distance, idx = tree.query(x)
    #     print(distance, idx)
    #





    from utils.running_stats import ZFilter
    from worker import Worker
    import json

    with open('config.json') as f:
        config = json.load(f)

    dataset_path = 'log-files/InvertedPendulum-v1/Sep-24_01_41LOG_BRANCH_WIDTH::4LOG_BRANCH_DEPTH::1'

    # dataset_path = 'log-files/InvertedPendulum-v1'  # 'log-files/InvertedPendulum-v1/Sep-23_12_38'  # 'log-files/InvertedPendulum-v1/Sep-22_13_29'
    worker = Worker(config, log_dir=dataset_path)

    # try:
    #     worker.imagination.load()
    # except Exception as e:
    tf.logging.info('Pre training model')
    worker.imagination.pretrain_model(data_dir=dataset_path + '/dataset',
                                      concat_frames=config['CONCATENATE_FRAMES'])

    ob_filter = ZFilter((worker.env_dim[0],))

    TEST_RUNS = 1000
    worker.warmup(ob_filter, 60, history_depth=config['CONCATENATE_FRAMES'])
    # worker.imagination.reset(ob1)
    losses_v = []
    obs = []
    losses_v = 0
    losses_r = 0
    obs = []
    ep_loss_v = []
    ep_loss_r = []
    confs = []
    scores = []
    depths_means = []
    depths_vars = []

    from collections import deque

    history = deque(maxlen=config["CONCATENATE_FRAMES"])

    for _ in range(TEST_RUNS):
        done = False
        ob = worker.env.reset()
        history.append(ob)
        while not done:
            # worker.env.render()
            history.append(ob)
            h = ob_filter(np.array(history).flatten())
            worker.imagination.set_state(h)

            act, v, score, depths = worker.explore_options(world_state=h, n_branches=2)
            depths = np.array(depths)

            depths_means.append(depths.mean())
            depths_vars.append(depths.var())

            ob1, r, done, _ = worker.env.step(act)

            # obs.append((ob1_tilde[1] - ob1[1]))
            v = worker.agent.get_value(h)
            # loss_r = 0.5 * np.square(reward_tilde - r)

            v_tilde = worker.agent.get_value(h)

            loss_v = 0.5 * np.square(v_tilde - v)
            losses_v += loss_v
            # losses_r += loss_r

            ob = ob1.copy()

        ep_loss_v.append(losses_v)
        ep_loss_r.append(losses_r)
        losses_r = losses_v = 0

    tf.logging.info('Eval ended {}, {}'.format(np.mean(ep_loss_r), np.mean(ep_loss_v)))

    plt.plot(depths_means)
    plt.savefig("mean_depth")
    plt.close()
    plt.plot(depths_vars)
    plt.savefig("var_depth")
    plt.close()
    plt.plot(confs)
    plt.savefig("conf")
    plt.close()
    plt.plot(ep_loss_v)
    plt.savefig("done")
    plt.close()
    plt.plot(ep_loss_v)
    plt.savefig("ep_loss-v")
    plt.close()
    plt.plot(ep_loss_r)
    plt.savefig("ep_loss_r")
