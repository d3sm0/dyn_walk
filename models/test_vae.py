import glob
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

from memory.dataset import Dataset
from models.cvae import VAE


def main():
    data = prepare_data(load_data())
    dataset = Dataset(data=data , batch_size=64 , shuffle=True)

    vae = VAE(obs_dim=17 , acts_dim=6 , latent_dim=2 , batch_size=64)
    losses = []
    kls = []
    now = datetime.utcnow().strftime("%b-%d_%H_%M_%S")  # create unique dir
    main_path = os.path.join('logs' , now)
    os.makedirs(main_path)
    writer = tf.summary.FileWriter(main_path)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    idx = 0
    for _ in range(2):
        for batch in dataset.iterate_once():
            loss , kl , _ , summary = vae.sess.run([vae.img_loss , vae.kl , vae.train , vae.summary], feed_dict={
                vae.obs: batch['obs'] ,
                vae.obs1: batch['obs1'] ,
                vae.acts: batch['acts']
            })
            writer.add_summary(summary=summary , global_step=idx)
            writer.flush()
            losses.append(loss)
            kls.append(kl)
            idx = +1
    print(np.mean(losses), np.mean(kl))
    saver.save(sess = vae.sess, save_path=os.path.join(main_path, 'model.ckpt'))
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.plot(kls)
    # plt.show()
    plt.savefig('cvae_sa_log_var.png')
    plt.close()


def load_data():
    try:
        with open("logs/dataset.pkl" , "rb") as fin:
            datasets = pickle.load(fin)
    except IOError:
        datasets = {}
        names = list(glob.glob("ppo-logs/dataset/dump_5*0"))
        i = 0
        for file_name in names:
            i += 1
            print(file_name , i , len(names) , i / len(names))
            with open(file_name , "rb") as fin:
                data = pickle.load(fin , encoding="bytes")
                if len(datasets) == 0:
                    for k , v in data.items():
                        datasets[k] = np.array(v)
                else:
                    for k , v in data.items():
                        if np.ndim(v) == 0:
                            datasets[k] = np.append(datasets[k] , v)
                        else:
                            datasets[k] = np.concatenate((datasets[k] , v) , axis=0)
                            # with open("log-files/dataset.pkl" , "wb") as fout:
                            #     pickle.dump(datasets , fout)
    return datasets


def prepare_data(datasets):
    obs0 = datasets[b'obs'][:-1]
    obs1 = datasets[b'obs'][1:]
    acts = datasets[b'acts'][:-1]
    rws = datasets[b'rws'][:-1]
    dones = datasets[b'ds'][:-1]

    # tmp1 = []
    # tmp2 = []
    # tmp3 = []
    # tmp4 = []
    # obss , obss1 , actss , rwss = [] , [] , [] , []
    #
    # for d , o , o1 , a , r in zip(dones , obs0 , obs1 , acts , rws):
    #     if d == 1:
    #         obss.append(np.copy(tmp1))
    #         obss1.append(np.copy(tmp2))
    #         actss.append(np.copy(tmp3))
    #         rwss.append(np.copy(tmp4))
    #         tmp1 , tmp2 , tmp3 , tmp4 = [] , [] , [] , []
    #     else:
    #         tmp1.append(o)
    #         tmp2.append(o1)
    #         tmp3.append(a)
    #         tmp4.append(r)
    return {
        'obs': obs0 ,
        'obs1': obs1 ,
        'acts': acts ,
        'rws': rws
    }


if __name__ == "__main__":
    main()
