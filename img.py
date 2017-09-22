import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import glob
from utils.tf_utils import _load , _save
from memory.dataset import Dataset
import numpy as np
from models.cvae import VAE
from models.fc_model import FCModel
from os.path import join
from utils.misc_utils import load_data, train_test_set

class Imagination(object):
    def __init__(self , obs_dim , acts_dim , model_path=None , model='vae' , z_dim=None):
        self.s , self.s1 , self.a = [] , [] , []
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.is_trained = False
        self.model_name = model
        if self.model_name == 'vae':
            assert z_dim is not None
            self.model = VAE(obs_dim=obs_dim , acts_dim=acts_dim , z_dim=z_dim , batch_size=64)
        elif self.model_name == 'fc':
            self.model = FCModel(obs_dim=obs_dim , acts_dim=acts_dim)

        self.model_path = join(model_path , '{}.ckpt'.format(self.model_name))

    def set_state(self , state):
        self.state = state
        return self.state

    def step(self , a):
        # s1_tilde = img.sess.run(img.model.s1_tilde, feed_dict={img.model.sa: sa})
        # r_tilde = img.calc_reward(ob, act, s1_tilde)

        s1_tilde = np.squeeze(self.model.step(self.state , a))
        # sudgested by manuel
        reward , done = self.calc_reward(self.state , a , s1_tilde)
        self.state = s1_tilde

        return s1_tilde , reward , done , {}

    def collect(self , s , a , s1):
        self.s.append(s)
        self.a.append(a)
        self.s1.append(np.array(s1))
        # TODO if somehting do training
        # if self.max_len == len(self.sa):
        #     print("imagination train loss:" , self.train())
        #     self.sa = []
        #     self .s1 = []
        # self.train()

    def train(self , iter_batch=10 , dataset=None , verbose=False):
        losses = []
        i = 0
        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs'][:-1]
                s1 = batch['obs'][1:]
                acts = batch['acts'][:-1]
                stats = self.model.train(s0 , acts , s1)
                losses.append(stats)
                i += 1
                if verbose and i % 100 == 0:
                    tf.logging.info('Dataset coverd {}. current loss {}'.format(i // dataset.n , stats))
        return {'avg_loss': np.mean(losses)}

    @staticmethod
    def calc_reward(s0 , a , s1):
        # reward = 1.0
        # ob = s0
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone

        posbefore = s0[0]
        posafter , height , ang = s1[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        return reward , done

    def eval_model(self , dataset , iter_batch=1 , verbose=False):
        losses = []
        i = 0
        for _ in range(iter_batch):
            for batch in dataset.iterate_once():
                s0 = batch['obs'][:-1]
                s1 = batch['obs'][1:]
                acts = batch['acts'][:-1]
                stats = self.model.eval(s0 , acts , s1)
                i += 1
                losses.append(stats)
                if i % 100 == 0:
                    tf.logging.info('Dataset coverd {}. current loss {}'.format(i // dataset.n , stats))
        return {'avg_loss': np.mean(losses)}

    def pretrain_model(self , data_dir, verbose = True):

        data = load_data(data_dir=data_dir)
        assert data is not None
        train_set , test_set = train_test_set(datasets=data)

        train_loss = self.train(
            dataset=Dataset(data={'obs': train_set[0] , 'acts': train_set[1] , 'obs1': train_set[2]} , shuffle=True) ,
            iter_batch=1 , verbose=verbose)

        tf.logging.info('Testing results. Train loss: {}'.format(train_loss))
        test_loss = self.eval_model(
            dataset=Dataset(data={'obs': test_set[0] , 'acts': test_set[1] , 'obs1': test_set[2]} , shuffle=True) ,
            iter_batch=1 , verbose=verbose)
        tf.logging.info('Pre train finished. Test loss: {}'.format(test_loss))

        _save(saver=self.model.saver , sess=self.model.sess , log_dir=self.model_path)
        self.is_trained = True

        return {'train_loss': train_loss , 'test_loss': test_loss}



if __name__ == "__main__":

    from utils.running_stats import ZFilter
    from worker import Worker
    import json

    with open('config.json') as f:
        config = json.load(f)

    dataset_path = 'log-files/InvertedPendulum-v1/Sep-22_13_29'
    worker = Worker(config , log_dir=dataset_path)
    # _load(sess = worker.agent.sess, saver = worker.agent.saver,log_dir='log-files/old_logs/Sep-19_01_19')
    img = Imagination(obs_dim=worker.env_dim[0] , acts_dim=worker.env_dim[1] , model_path='tf-models/' , model='vae' ,
                      z_dim=2)

    # dataset = Imagination.build_dataset(log_dir=dataset_path + '/dataset')

    try:
        _load(saver=img.model.saver , sess=img.model.sess , log_dir=img.model_path)
    except Exception as e:
        tf.logging.info('Pre training model')
        img.pretrain_model(data_dir=dataset_path + '/dataset')

    ob_filter = ZFilter((worker.env_dim[0] ,))

    TEST_RUNS = 1000
    worker.warmup(ob_filter , 60)
    # img.reset(ob1)
    losses_v = []

    losses_v = 0
    losses_r = 0
    ep_loss_v = []
    ep_loss_r = []
    for _ in range(TEST_RUNS):
        done = False
        ob1 = worker.env.reset()
        ob = ob_filter(ob1)
        while not done:
            ob = ob_filter(ob1)

            act , _ = worker.agent.get_action_value(ob)

            img.set_state(ob)  # follow the ground truth
            ob1_tilde , reward_tilde , done , _ = img.step(act)

            ob1 , r , done , _ = worker.env.step(act)

            loss_r = 0.5 * np.square(reward_tilde - r)

            v_tilde = worker.agent.get_value([ob1_tilde])
            v = worker.agent.get_value([ob1])
            loss_v = 0.5 * np.square(v_tilde - v)
            losses_v += loss_v
            losses_r += loss_r
        ep_loss_v.append(losses_v)
        ep_loss_r.append(losses_r)
        losses_r = losses_v = 0
    plt.plot(ep_loss_v)
    plt.savefig("ep_loss-v")
    plt.close()
    plt.plot(ep_loss_r)
    plt.savefig("ep_loss_r")
