import gym
from value import ValueNetwork
import tensorflow as tf

env = gym.make('Walker2d-v1')
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

value = ValueNetwork(obs_dim)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
s = env.reset()
v_hat =  value.vf.eval(feed_dict = {value.obs:[s]})
print(v_hat)
a = env.action_space.sample()
s1, r, t, _  =env.step(a)
v_hat_2 = value.vf.eval(feed_dict = {value.obs:[s1]})
feed_dict ={
    value.obs:[s],
    value.tdl:[r]
}
l, _ = sess.run([value.loss, value.train], feed_dict=feed_dict)

assert l >= 0
print (l)

for grad in value.grads:
    g = grad.eval(feed_dict=feed_dict)
    print(g.mean(), g.max(), g.min())

