import tensorflow as tf
import tensorflow_probability as tfp
from agent import Agent
import numpy as np


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = []
    for rewards in all_rewards:
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        # Determine reward of each step depending on following rewards using discount_rate
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        # Save list of discounted rewards in all_discounted_rewards
        all_discounted_rewards.append(discounted_rewards)
    # Make all_discounted_rewards one dimensional
    flat_rewards = np.concatenate(all_discounted_rewards)
    # Get mean and standard deviation of all_discounted_rewards
    rewards_mean = flat_rewards.mean()
    rewards_std = flat_rewards.std()
    # Return normalized version of all_discounted_rewards
    return [(discounted_rewards - rewards_mean) / rewards_std for discounted_rewards in all_discounted_rewards]


def multinomial(probs, invalid=[]):
    # Remove invalid actions from possible actions
    actions = list(set(range(max(2, probs.shape[1]))) - set(invalid))
    # Two possible actions and only one action is valid: select that one
    if probs.shape[1] == 1 and len(actions) == 1:
        action = actions[0]
        y_target = tf.constant([actions])
    # Two possible actions and both are valid: randomly select action by comparing to random float (between 0 and 1)
    elif probs.shape[1] == 1:
        y_target = tf.random.uniform([1, 1]) < probs
        action = int(y_target.numpy()[0, 0])
    # More than two possible actions: use multinomial selection
    else:
        probs_c = np.copy(probs)
        # If there are invalid actions: set their probability to be selected to 0
        if probs.shape[1] > len(actions):
            probs_c[0, invalid] = 0.
            probs_c = probs_c / np.sum(probs_c)
        y_target = tfp.distributions.Multinomial(1., probs=probs_c[0]).sample(1)
        action = np.argmax(y_target.numpy()[0])
    return action, tf.cast(y_target, tf.float32)


class PG(Agent):

    def __init__(self, layers, loss_fn, optimizer, discount_factor, file=None):
        # Create model by calling Agent class initializer
        super(PG, self).__init__(layers, loss_fn, optimizer, discount_factor, file)

    def run_policy(self, obs, invalid=[]):
        with tf.GradientTape() as tape:
            # Get model output for observation as input
            pred = self.model(obs[np.newaxis])
            # Select action depending on its output probability
            action, y_target = multinomial(pred, invalid)
            # Calculate loss with one-hot encoded action as target
            loss = tf.reduce_mean(self.loss_fn(y_target, pred))
        # Return action as integer and the gradient that would make it more likely
        return action, tape.gradient(loss, self.model.trainable_variables)

    def apply_grads(self, all_rewards, all_grads, discount=True):
        # Get discounted and normalized rewards
        if discount:
            all_rewards = discount_and_normalize_rewards(all_rewards, self.discount_factor)
        # Calculate weighted mean of gradients by multiplying with rewards
        all_mean_grads = []
        for var_i in range(len(self.model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[ep_i][step][var_i]
                 for ep_i, rewards in enumerate(all_rewards)
                  for step, final_reward in enumerate(rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        # Apply mean gradients to model
        self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))
