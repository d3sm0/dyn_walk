import gym
from policy import PolicyNetwork
import tensorflow as tf
env = gym.make('Walker2d-v1')

act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

policy = PolicyNetwork(obs_dim , act_dim)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
s = env.reset()
a = policy.pi.sample().eval(feed_dict = {policy.obs:[s]}).flatten()
adv = 4
assert a.shape[0] == act_dim
mu_old = policy.mu.eval(feed_dict = {policy.obs:[s]})
logstd_old= policy.logstd.eval()
kl = policy.kl.eval(feed_dict={policy.obs:[s], policy.acts:[a], policy.mu_old:mu_old, policy.logstd_old:logstd_old})
assert kl.mean() == 0
feed_dict = {
    policy.obs:[s],
    policy.acts:[a],
    policy.adv:[adv],
    policy.mu_old:mu_old,
    policy.logstd_old:logstd_old
}
for l in policy.losses:
    print(l.eval(feed_dict = feed_dict), '\n')

