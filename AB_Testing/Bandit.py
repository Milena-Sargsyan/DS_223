"""
Homework 2: A/B Testing with Multi-Armed Bandits

This module implements two bandit algorithms:
  - Epsilon-Greedy with decaying epsilon (1/t)
  - Thompson Sampling with known precision

Bandit rewards: [1, 2, 3, 4]
Number of trials: 20000

Author: Milena Sargsyan
Date: 2026

Constants:
    BANDIT_REWARDS (list[int]): True mean rewards for each bandit arm.
    NUM_TRIALS (int): Total number of experiment steps.
"""


from abc import ABC, abstractmethod
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

np.random.seed(42)

# Defining constants
BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000

# ---
# Abstract Base Class
# ---


class Bandit(ABC):
    """Abstract base class for a multi-armed bandit."""

    # ==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """
        :param p: True mean reward of this bandit arm.
        :type p: float
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """Sample reward from this arm."""
        pass

    @abstractmethod
    def update(self, x):
        """Update the bandit's internal statistics after observing reward x.

        :param x: Observed reward from the pulled arm.
        :type x: float
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the full bandit experiment for NUM_TRIALS steps.

        :returns: Tuple of (results, rewards, regrets) where results is a list
                  of (bandit_index, reward) tuples, rewards is a list of per-trial
                  rewards, and regrets is a list of per-trial regrets.
        :rtype: tuple
        """
        pass

    @abstractmethod
    def report(self, results, algorithm):
        """Save results to CSV, print cumulative reward and regret.

        :param results: List of (bandit_index, reward) tuples from the experiment.
        :type results: list[tuple]
        :param algorithm: Name of the algorithm (used as a CSV column value).
        :type algorithm: str
        """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

# --------------------------------------#
# Visualization


class Visualization():
    """Produces all required plots for the experiment."""

    def plot1(self, rewards_eg, rewards_ts):
        """
        Visualize the learning process for each algorithm on linear and log scale.

        :param rewards_eg: Per-trial rewards from Epsilon-Greedy.
        :param rewards_ts: Per-trial rewards from Thompson Sampling.
        """
        for rewards, name in [(rewards_eg, "Epsilon-Greedy"),
                              (rewards_ts, "Thompson Sampling")]:
            cumulative = np.cumsum(rewards)

            # Linear scale
            plt.figure(figsize=(10, 5))
            plt.plot(cumulative)
            plt.title(f"{name} - Cumulative Reward (Linear Scale)")
            plt.xlabel("Trial")
            plt.ylabel("Cumulative Reward")
            plt.savefig(f"{name.replace(' ', '_')}_linear.png")
            plt.show()

            # Log scale
            plt.figure(figsize=(10, 5))
            plt.plot(cumulative)
            plt.xscale("log")
            plt.title(f"{name} - Cumulative Reward (Log Scale)")
            plt.xlabel("Trial (log)")
            plt.ylabel("Cumulative Reward")
            plt.savefig(f"{name.replace(' ', '_')}_log.png")
            plt.show()

    def plot2(self, rewards_eg, rewards_ts, regrets_eg, regrets_ts):
        """
        Compare Epsilon-Greedy and Thompson Sampling cumulative rewards and regrets.

        :param rewards_eg: Per-trial rewards from Epsilon-Greedy.
        :param rewards_ts: Per-trial rewards from Thompson Sampling.
        :param regrets_eg: Per-trial regrets from Epsilon-Greedy.
        :param regrets_ts: Per-trial regrets from Thompson Sampling.
        """
        # Cumulative rewards comparison
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(rewards_eg), label="Epsilon-Greedy")
        plt.plot(np.cumsum(rewards_ts), label="Thompson Sampling")
        plt.title("Cumulative Rewards: E-Greedy vs Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.savefig("comparison_rewards.png")
        plt.show()

        # Cumulative regrets comparison
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(regrets_eg), label="Epsilon-Greedy")
        plt.plot(np.cumsum(regrets_ts), label="Thompson Sampling")
        plt.title("Cumulative Regrets: E-Greedy vs Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.savefig("comparison_regrets.png")
        plt.show()

# --------------------------------------#
# Epsilon Greedy


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit with decaying exploration rate epsilon = 1/t.

    :param p: True mean reward of this arm.
    :type p: float
    """

    def __init__(self, p):
        self.p = p             # true mean reward
        self.p_estimate = 0.0  # estimated mean reward
        self.N = 0             # number of times this arm was pulled

    def __repr__(self):
        return f"EpsilonGreedy(true_mean={self.p}, estimate={self.p_estimate:.4f}, pulls={self.N})"

    def pull(self):
        """Sample reward from N(p, 1)."""
        return np.random.randn() + self.p

    def update(self, x):
        """Update running mean estimate with new observation x."""
        self.N += 1
        self.p_estimate += (x - self.p_estimate) / self.N

    def experiment(self):
        """
        Run Epsilon-Greedy experiment. Epsilon decays as 1/t.

        :returns: (results, rewards, regrets)
        """
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        best_reward = max(BANDIT_REWARDS)

        rewards = []
        regrets = []
        results = []

        for t in range(1, NUM_TRIALS + 1):
            epsilon = 1.0 / t  # decaying epsilon

            if np.random.random() < epsilon:
                chosen = np.random.randint(len(bandits))  # explore
            else:
                chosen = np.argmax([b.p_estimate for b in bandits])  # exploit

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            rewards.append(reward)
            regrets.append(best_reward - BANDIT_REWARDS[chosen])
            results.append((chosen + 1, reward))

        logger.info("Epsilon-Greedy experiment completed.")
        return results, rewards, regrets

    def report(self, results, algorithm="Epsilon-Greedy"):
        """
        Save results to CSV and log cumulative reward and regret.

        :param results: List of (bandit_index, reward) tuples.
        :param algorithm: Algorithm name for the CSV column.
        """
        csv_path = "results.csv"
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for bandit_idx, reward in results:
                writer.writerow([bandit_idx, round(reward, 6), algorithm])

        total_reward = sum(r for _, r in results)
        total_regret = NUM_TRIALS * max(BANDIT_REWARDS) - total_reward

        logger.info(f"[{algorithm}] Cumulative Reward: {total_reward:.2f}")
        logger.info(f"[{algorithm}] Cumulative Regret: {total_regret:.2f}")


# --------------------------------------#
# Thompson Sampling


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit using a Gaussian model with known precision.

    Maintains a Normal posterior over each arm's mean and samples from it
    to decide which arm to pull.

    :param p: True mean reward of this arm.
    :type p: float
    :param precision: Known observation precision (default 1.0).
    :type precision: float
    """

    def __init__(self, p, precision=1.0):
        self.p = p
        self.precision = precision  # known likelihood precision

        # Prior: N(0, 1)
        self._prior_mean = 0.0
        self._prior_prec = 1.0

        # Posterior parameters (updated after each observation)
        self._post_mean = self._prior_mean
        self._post_prec = self._prior_prec

        self.N = 0  # number of pulls
        self._sum_x = 0.0  # running sum of observed rewards

    def __repr__(self):
        return f"ThompsonSampling(true_mean={self.p}, post_mean={self._post_mean:.4f}, pulls={self.N})"

    def pull(self):
        """Sample reward from N(p, 1/precision)."""
        return np.random.randn() / np.sqrt(self.precision) + self.p

    def update(self, x):
        """Update Gaussian posterior after observing reward x."""
        self.N += 1
        self._sum_x += x
        self._post_prec = self._prior_prec + self.N * self.precision
        self._post_mean = (self._prior_prec * self._prior_mean + self.precision * self._sum_x) / self._post_prec

    def sample(self):
        """Draw a sample from the current posterior."""
        return np.random.randn() / np.sqrt(self._post_prec) + self._post_mean

    def experiment(self):
        """
        Run Thompson Sampling experiment over NUM_TRIALS trials.

        :returns: (results, rewards, regrets)
        """
        bandits = [ThompsonSampling(p) for p in BANDIT_REWARDS]
        best_reward = max(BANDIT_REWARDS)

        rewards = []
        regrets = []
        results = []

        for _ in range(NUM_TRIALS):
            chosen = np.argmax([b.sample() for b in bandits])

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            rewards.append(reward)
            regrets.append(best_reward - BANDIT_REWARDS[chosen])
            results.append((chosen + 1, reward))

        logger.info("Thompson Sampling experiment completed.")
        return results, rewards, regrets

    def report(self, results, algorithm="Thompson Sampling"):
        """
        Save results to CSV and log cumulative reward and regret.

        :param results: List of (bandit_index, reward) tuples.
        :param algorithm: Algorithm name for the CSV column.
        """
        csv_path = "results.csv"
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for bandit_idx, reward in results:
                writer.writerow([bandit_idx, round(reward, 6), algorithm])

        total_reward = sum(r for _, r in results)
        total_regret = NUM_TRIALS * max(BANDIT_REWARDS) - total_reward

        logger.info(f"[{algorithm}] Cumulative Reward: {total_reward:.2f}")
        logger.info(f"[{algorithm}] Cumulative Regret: {total_regret:.2f}")


# ---
# Comparison

def comparison():
    """Run both experiments and produce all visualizations and reports."""
    # Remove old CSV to avoid appending to stale results
    if os.path.exists("results.csv"):
        os.remove("results.csv")

    # Epsilon-Greedy
    eg = EpsilonGreedy(BANDIT_REWARDS[0])
    res_eg, rewards_eg, regrets_eg = eg.experiment()
    eg.report(res_eg, algorithm="Epsilon-Greedy")

    # Thompson Sampling
    ts = ThompsonSampling(BANDIT_REWARDS[0])
    res_ts, rewards_ts, regrets_ts = ts.experiment()
    ts.report(res_ts, algorithm="Thompson Sampling")

    # Visualize
    viz = Visualization()
    viz.plot1(rewards_eg, rewards_ts)
    viz.plot2(rewards_eg, rewards_ts, regrets_eg, regrets_ts)


if __name__ == '__main__':
    logger.info("Starting A/B Testing experiment with Multi-Armed Bandits.")
    comparison()
    logger.info("Experiment complete. Results saved to results.csv.")
