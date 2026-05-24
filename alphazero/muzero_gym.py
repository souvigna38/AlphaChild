"""MuZero MCTS and training loop for single-agent Gym environments."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class Node:
    def __init__(self, prior: float = 0):
        self.prior = prior
        self.hidden_state = None
        self.reward = 0.0
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MuZeroGymMCTS:
    """MCTS that plans with the learned dynamics model (single-agent, no value sign flip)."""

    def __init__(self, game, model, args: dict[str, Any]):
        self.game = game
        self.model = model
        self.args = args

    def get_ucb_score(self, parent: Node, child: Node) -> float:
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = 1 - ((child.value() + 1) / 2)
        return q_value + self.args["C"] * (math.sqrt(parent.visit_count) / (child.visit_count + 1)) * child.prior

    def select_child(self, node: Node) -> tuple[int, Node]:
        best_score = -np.inf
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            score = self.get_ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
        encoded = self.game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        hidden_state, policy_logits, _ = self.model.initial_inference(tensor)

        root = Node()
        root.hidden_state = hidden_state
        root.visit_count = 1

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size
        )

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        if policy.sum() > 0:
            policy /= policy.sum()

        for action in range(self.game.action_size):
            if policy[action] > 0:
                root.children[action] = Node(prior=policy[action])

        gamma = self.args.get("gamma", 0.99)

        for _ in range(self.args["num_searches"]):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            action_tensor = torch.tensor([action], dtype=torch.long)
            hidden_state, reward, policy_logits, value = self.model.recurrent_inference(
                parent.hidden_state, action_tensor
            )

            node.hidden_state = hidden_state
            node.reward = reward.item()
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            bootstrap = value.item()

            for a in range(self.game.action_size):
                if policy[a] > 0:
                    node.children[a] = Node(prior=policy[a])

            for bnode in reversed(search_path):
                bnode.visit_count += 1
                bnode.value_sum += bootstrap
                bootstrap = bnode.reward + gamma * bootstrap

        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        return action_probs


class MuZeroGym:
    def __init__(self, model, optimizer, game, args: dict[str, Any]):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MuZeroGymMCTS(game, model, args)

    def self_play(self) -> dict[str, list]:
        observations: list = []
        actions: list = []
        policies: list = []
        rewards: list = []

        state = self.game.get_initial_state()
        episode_return = 0.0

        while True:
            action_probs = self.mcts.search(state)
            observations.append(self.game.get_encoded_state(state))
            policies.append(action_probs)

            action = int(np.random.choice(self.game.action_size, p=action_probs))
            actions.append(action)

            state = self.game.get_next_state(state, action)
            reward, done = self.game.get_value_and_terminated(state, action)
            rewards.append(reward)
            episode_return += reward

            if done:
                values = []
                bootstrap = 0.0
                for r in reversed(rewards):
                    bootstrap = r + self.args.get("gamma", 0.99) * bootstrap
                    values.append(bootstrap)
                values.reverse()

                return {
                    "observations": observations,
                    "actions": actions,
                    "policies": policies,
                    "values": values,
                    "rewards": rewards,
                    "episode_return": episode_return,
                }

    def train(self, replay_buffer: list) -> None:
        random.shuffle(replay_buffer)
        k_steps = self.args["num_unroll_steps"]

        for batch_start in range(0, len(replay_buffer), self.args["batch_size"]):
            batch_games = replay_buffer[batch_start : batch_start + self.args["batch_size"]]
            if len(batch_games) < 1:
                continue

            batch_obs = []
            batch_actions = []
            batch_target_policies = []
            batch_target_values = []
            batch_target_rewards = []

            for game_hist in batch_games:
                game_len = len(game_hist["observations"])
                pos = np.random.randint(game_len)

                batch_obs.append(game_hist["observations"][pos])
                step_actions = []
                step_policies = []
                step_values = []
                step_rewards = []

                for k in range(k_steps + 1):
                    idx = pos + k
                    if idx < game_len:
                        step_policies.append(game_hist["policies"][idx])
                        step_values.append(game_hist["values"][idx])
                        step_rewards.append(game_hist["rewards"][idx])
                        step_actions.append(game_hist["actions"][idx] if idx < len(game_hist["actions"]) else 0)
                    else:
                        step_policies.append(np.ones(self.game.action_size) / self.game.action_size)
                        step_values.append(0.0)
                        step_rewards.append(0.0)
                        step_actions.append(0)

                batch_actions.append(step_actions)
                batch_target_policies.append(step_policies)
                batch_target_values.append(step_values)
                batch_target_rewards.append(step_rewards)

            obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
            hidden_state, pred_policy, pred_value = self.model.initial_inference(obs_tensor)

            target_policy_0 = torch.tensor(np.array([tp[0] for tp in batch_target_policies]), dtype=torch.float32)
            target_value_0 = torch.tensor(np.array([[tv[0]] for tv in batch_target_values]), dtype=torch.float32)

            total_loss = F.cross_entropy(pred_policy, target_policy_0) + F.mse_loss(pred_value, target_value_0)

            for k in range(k_steps):
                actions_k = torch.tensor([ba[k] for ba in batch_actions], dtype=torch.long)
                hidden_state, pred_reward, pred_policy, pred_value = self.model.recurrent_inference(
                    hidden_state, actions_k
                )

                target_policy_k = torch.tensor(
                    np.array([tp[k + 1] for tp in batch_target_policies]), dtype=torch.float32
                )
                target_value_k = torch.tensor(
                    np.array([[tv[k + 1]] for tv in batch_target_values]), dtype=torch.float32
                )
                target_reward_k = torch.tensor(
                    np.array([[tr[k]] for tr in batch_target_rewards]), dtype=torch.float32
                )

                hidden_state.register_hook(lambda grad: grad * 0.5)
                total_loss += (
                    F.cross_entropy(pred_policy, target_policy_k)
                    + F.mse_loss(pred_value, target_value_k)
                    + F.mse_loss(pred_reward, target_reward_k)
                )

            total_loss /= k_steps + 1
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def learn(self, progress_callback=None) -> None:
        for iteration in range(self.args["num_iterations"]):
            replay_buffer = []
            self.model.eval()
            for _ in range(self.args["num_selfPlay_iterations"]):
                replay_buffer.append(self.self_play())

            self.model.train()
            for _ in range(self.args["num_epochs"]):
                self.train(replay_buffer)

            if progress_callback:
                progress_callback(iteration, replay_buffer)

            torch.save(self.model.state_dict(), f"muzero_gym_model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"muzero_gym_optimizer_{iteration}.pt")
