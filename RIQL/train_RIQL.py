import torch
from torch.utils.data import DataLoader
from RIQL import V_RIQL, Q_RIQL, Policy_RIQL, Policy_RIQL_Normal, Policy_RIQL_Det
from dataset import Dataset
import math
import copy
import wandb
import gym
import datetime
import argparse
import glob
import os
import sys
from tqdm import tqdm

def Q_alpha(q_riql, s: torch.Tensor, a: torch.Tensor, alpha: float) -> torch.Tensor:
    qs = torch.stack([q(s, a).squeeze(-1) for q in q_riql], dim=0).transpose(1, 0)
    return torch.quantile(qs, alpha, dim=1, keepdim=True)


def L_2_tau(x: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(x >= 0, tau, 1 - tau)
    return weight * x.pow(2)

def huber_loss(x: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(torch.abs(x) <= delta, 1 / delta * 0.5 * x.pow(2),  (torch.abs(x) - 0.5 * delta))
    
def V_loss(q_alpha, v_riql, s, tau):
    losses = L_2_tau(q_alpha - v_riql(s), tau=tau)
    return torch.mean(losses)

def Q_loss(q_riql, v_target, s, a, delta):
    q_vals = torch.stack([q(s, a) for q in q_riql], dim=1)
    v_expanded = v_target.unsqueeze(1).expand_as(q_vals)
    losses = huber_loss(v_expanded - q_vals, delta=delta).mean(dim=0).sum()
    
    return losses

def policy_loss(q_alpha, v_next, policy_riql, s, a, beta):
    with torch.no_grad():
        A = q_alpha - v_next
        W = torch.exp(beta * A).clamp(max=100.0)
    losses = W * policy_riql.log_prob(s, a)
    return -torch.mean(losses)

def train_RIQL_epoch(q_riql, v_riql, q_riql_target, v_riql_target, policy_riql, dataloader, alpha, tau, gamma, delta, beta, optimizers, device):
    for q in q_riql:
        q.train()
    v_riql.train()
    policy_riql.train()
    v_optimizer, q_optimizers, policy_optimizer = optimizers

    v_losses = []
    q_losses = []
    policy_losses = []
    td_target_means = []

    polyak = 0.005  # target network update rate

    for step, (s, a, r, s_next, done) in enumerate(dataloader):

        v_optimizer.zero_grad()
        for q_optimizer in q_optimizers:
            q_optimizer.zero_grad()
        policy_optimizer.zero_grad()
        
        # move data to device
        s = s.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True).unsqueeze(-1)
        s_next = s_next.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        
        with torch.no_grad():
            q_alpha = Q_alpha(q_riql_target, s, a, alpha)
            v_next = v_riql(s)
            v_target = r + gamma * v_riql_target(s_next) * (1.0 - done.float()).unsqueeze(-1)
        v_loss = V_loss(q_alpha, v_riql, s, tau)
        q_loss = Q_loss(q_riql, v_target, s, a, delta=delta) 
        policy_loss_value= policy_loss(q_alpha, v_next, policy_riql, s, a, beta)
        
        total_loss = v_loss + q_loss + policy_loss_value
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(v_riql.parameters(), max_norm=1.0)
        for q in q_riql:
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(policy_riql.parameters(), max_norm=1.0)
        v_optimizer.step()
        for q_optimizer in q_optimizers:
            q_optimizer.step()
        policy_optimizer.step()

        v_losses.append(v_loss.item())
        q_losses.append(q_loss.item())
        policy_losses.append(policy_loss_value.item())
        
        # target network update
        with torch.no_grad():
            for v_target_param, v_param in zip(v_riql_target.parameters(), v_riql.parameters()):
                v_target_param.data.copy_(polyak * v_param.data + (1 - polyak) * v_target_param.data)
            for q_target, q in zip(q_riql_target, q_riql):
                for q_target_param, q_param in zip(q_target.parameters(), q.parameters()):
                    q_target_param.data.copy_(polyak * q_param.data + (1 - polyak) * q_target_param.data)
            td_target_means.append(torch.mean(v_target).item())

    # mean losses
    mean_v_loss = sum(v_losses) / len(v_losses)
    mean_q_loss = sum(q_losses) / len(q_losses)
    mean_policy_loss = sum(policy_losses) / len(policy_losses)
    mean_td_target_mean = sum(td_target_means) / len(td_target_means)
    return mean_v_loss, mean_q_loss, mean_policy_loss, mean_td_target_mean
    
def evaluate_policy(
    policy,
    mean: torch.Tensor,
    std: torch.Tensor,
    env_name: str = "Hopper-v2",
    num_episodes: int = 10,
    max_steps: int = 1000,
    discount: float = 1.0,
    device: str = "cpu"
):
    policy.eval()
    env = gym.make(env_name)
    returns = []

    for _ in range(num_episodes):

        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

        total_reward = 0.0
        done = False
        steps = 0
        discount_factor = 1.0

        while not done and steps < max_steps:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            n_state = (state - mean) / std

            with torch.no_grad():
                action_tensor = policy.get_action(n_state)
            action = action_tensor.cpu().numpy().flatten()

            step_ret = env.step(action)
            if len(step_ret) == 4:
                obs, reward, done, info = step_ret
            else:
                obs, reward, done, truncated, info = step_ret
                done = done or truncated

            total_reward += discount_factor * reward
            discount_factor *= discount
            steps += 1

        returns.append(total_reward)

    env.close()
    return returns
def get_env_params(env_name = "Hopper-v2"):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    env.close()
    return state_dim, action_dim, max_action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for DataLoader")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save checkpoints")
    parser.add_argument("--env", type=str, required=True, help="testing env")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha hyperparam")
    parser.add_argument("--tau", type=float, default=0.7, help="tau hyperparam")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma hyperparam")
    parser.add_argument("--delta", type=float, default=1.0, help="delta hyperparam")
    parser.add_argument("--beta", type=float, default=3.0, help="beta hyperparam")
    
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for optimizers")
    parser.add_argument("--K", type=int, default=5, help="Number of Q networks")
    parser.add_argument("--env_name", type=str, required=True)
    args = parser.parse_args()
    run_id = os.path.basename(args.save_folder) +  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    
    wandb.init(project="RIQL", 
        mode="offline",
        config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "tau": args.tau,
        "gamma": args.gamma,
        "delta": args.delta,
        "beta": args.beta,
        "K": args.K
    },
        name=f"run_{run_id}")
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = Dataset(args.env)
    
    print(f"WandB directory: {wandb.run.dir}")
    os.makedirs(args.save_folder, exist_ok=True)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    epochs = config.epochs
    K = config.K
    lr = config.learning_rate
    alpha = config.alpha
    tau = config.tau
    gamma = config.gamma
    delta = config.delta
    beta = config.beta
    state_dim, action_dim, max_action = get_env_params(args.env_name)
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}, Max action: {max_action}")

    # load checkpoints if found
    checkpoint_files = glob.glob(f"{args.save_folder}/checkpoint_epoch_*.pth")
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        v_riql = V_RIQL(state_dim).to(device, non_blocking=True)
        q_riql = [Q_RIQL(state_dim, action_dim).to(device, non_blocking=True) for _ in range(K)]
        policy_riql = Policy_RIQL_Det(state_dim, action_dim, max_action).to(device, non_blocking=True)

        v_riql.load_state_dict(checkpoint["v_state_dict"])
        for q, q_state_dict in zip(q_riql, checkpoint["q_state_dicts"]):
            q.load_state_dict(q_state_dict)
        policy_riql.load_state_dict(checkpoint["policy_state_dict"])

        v_optimizer = torch.optim.Adam(v_riql.parameters(), lr=lr)
        q_optimizers = [torch.optim.Adam(q.parameters(), lr=lr) for q in q_riql]
        policy_optimizer = torch.optim.Adam(policy_riql.parameters(), lr=lr)

        v_optimizer.load_state_dict(checkpoint["v_optimizer_state_dict"])
        for q_optimizer, q_optimizer_state_dict in zip(q_optimizers, checkpoint["q_optimizer_state_dicts"]):
            q_optimizer.load_state_dict(q_optimizer_state_dict)
        policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        print(f"Resuming training from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        v_riql = V_RIQL(state_dim).to(device, non_blocking=True)
        q_riql = [Q_RIQL(state_dim, action_dim).to(device, non_blocking=True) for _ in range(K)]
        policy_riql = Policy_RIQL_Det(state_dim, action_dim, max_action).to(device, non_blocking=True)

        v_optimizer = torch.optim.Adam(v_riql.parameters(), lr=lr)
        q_optimizers = [torch.optim.Adam(q.parameters(), lr=lr) for q in q_riql]
        policy_optimizer = torch.optim.Adam(policy_riql.parameters(), lr=lr)
    v_riql_target = copy.deepcopy(v_riql).to(device, non_blocking=True)
    q_riql_target = [copy.deepcopy(q).to(device, non_blocking=True) for q in q_riql]
    v_riql_target.eval()
    for q in q_riql_target:
        q.eval()    
    optimizers = (v_optimizer, q_optimizers, policy_optimizer)
    obs_mean = dataset.mean.to(device, non_blocking=True)
    obs_std = dataset.std.to(device, non_blocking=True)
    for e in range(epochs):
        mean_v_loss, mean_q_loss, mean_policy_loss, mean_td_target_mean = train_RIQL_epoch(
            q_riql, v_riql, q_riql_target, v_riql_target, policy_riql, train_dataloader, 
            alpha, tau, gamma, delta, beta, optimizers, device)
        
        print(f"Epoch {e + 1} completed. v_loss: {mean_v_loss}, q_loss: {mean_q_loss}, policy_loss: {mean_policy_loss}, td_target_mean: {mean_td_target_mean}")
        
        log_dict = {
            "epoch": e + 1,
            "v_loss": mean_v_loss,
            "q_loss": mean_q_loss,
            "policy_loss": mean_policy_loss,
            "td_target_mean": mean_td_target_mean,
        }
        if (e + 1) % 10 == 0:
            returns = evaluate_policy(policy_riql, mean=obs_mean, std=obs_std, env_name=args.env_name, num_episodes=30, max_steps=1000, device=device)
            average_return = sum(returns) / len(returns)
            print(f"Average return over 30 episodes: {average_return}")
            log_dict["average_return"] = average_return

        sys.stdout.flush()
        wandb.log(log_dict)
        
        if (e + 1) % 10 == 0: # save checkpoint every 10 epochs
            checkpoint = {
            "epoch": e + 1,
            "policy_state_dict": policy_riql.state_dict(),
            "policy_optimizer_state_dict": policy_optimizer.state_dict(),
            "v_state_dict": v_riql.state_dict(),
            "v_optimizer_state_dict": v_optimizer.state_dict(),
            "q_state_dicts": [q.state_dict() for q in q_riql],
            "q_optimizer_state_dicts": [q_optimizer.state_dict() for q_optimizer in q_optimizers],
            "config": dict(config)
            }
            save_path = f"{args.save_folder}/checkpoint_epoch_{e + 1}.pth"
            torch.save(checkpoint, save_path)

            # keeping only the latest 3 checkpoints
            checkpoints = sorted(glob.glob(f"{args.save_folder}/checkpoint_epoch_*.pth"), key=os.path.getmtime)
            if len(checkpoints) > 3:
                for old_checkpoint in checkpoints[:-3]:
                    os.remove(old_checkpoint)

        # save the best checkpoint based on returns
        if (e + 1) % 10 == 0:
            average_return = sum(returns) / len(returns)
            if not hasattr(train_RIQL_epoch, "best_return") or average_return > train_RIQL_epoch.best_return:
                train_RIQL_epoch.best_return = average_return
                best_checkpoint = {
                "epoch": e + 1,
                "policy_state_dict": policy_riql.state_dict(),
                "policy_optimizer_state_dict": policy_optimizer.state_dict(),
                "v_state_dict": v_riql.state_dict(),
                "v_optimizer_state_dict": v_optimizer.state_dict(),
                "q_state_dicts": [q.state_dict() for q in q_riql],
                "q_optimizer_state_dicts": [q_optimizer.state_dict() for q_optimizer in q_optimizers],
                "config": dict(config),
                "best_return": train_RIQL_epoch.best_return
                }
                best_save_path = f"{args.save_folder}/best_checkpoint.pth"
                torch.save(best_checkpoint, best_save_path)
        
