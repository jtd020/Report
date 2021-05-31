from pud.dependencies import *
from pud.utils import set_global_seed, set_env_seed, AttrDict
import argparse

import cv2
import numpy as np

## Helper Function to get random maze. Mazes were generated using a seperate tool and stored in a file called random_mazes.txt
def get_rand_maze(location):
    random.seed(time.clock())
    maze = ""
    with open(location) as f:
        lines = f.readlines()
        maze = random.choice(lines)
        print(random.randint(0, len(lines)-1))
        print(maze)
    print(maze)
    maze = maze.strip()
    sidelen = int(np.sqrt(len(maze)))
    mazearr = []
    mazerow = []
    for i in range(len(maze)):
        mazerow.append(int(maze[i]))
        if ((i+1) % sidelen) == 0:
            mazearr.append(mazerow)
            mazerow = []

    return np.array(mazearr)

# def discretize_state(env, state, resolution=1.0):
#     (i, j) = np.floor(resolution * state).astype(np.int)
#     (height, width) = env.walls.shape
#     # Round down to the nearest cell if at the boundary.
#     if i == height:
#         i -= 1
#     if j == width:
#         j -= 1
#     return (i, j)


## Set Args to Train
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="config path",
                    type=str)
parser.add_argument("--checkpt", nargs='?', help="checkpoint directory",
                    type=str)
parser.add_argument("--train", help="train user",
                    type=bool)
parser.add_argument("--rand", help="generate random maze",
                    type=bool)
parser.add_argument("--sparse", help="generate random maze",
                    type=bool)
args = parser.parse_args()
train = args.train
print(train)
print(args.sparse)


## Config file
print("Opening Up Config File")
cfg_file = args.path
cfg = AttrDict(**eval(open(cfg_file, 'r').read()))

if args.checkpt is not None:
    cfg.ckpt_dir = args.checkpt
print(cfg.ckpt_dir)
print(cfg)
set_global_seed(cfg.seed)


## Create directory if it doesn't exist
if not os.path.exists(cfg.ckpt_dir):
    train = True
    os.makedirs(cfg.ckpt_dir)

rand_env = None
print(cfg.env.env_name)
if cfg.env.env_name == "RandMaze":
    rand_env = get_rand_maze("./random_mazes.txt")


## Load environment
print("Loading Env")
from pud.envs.simple_navigation_env import env_load_fn
env = env_load_fn(cfg.env.env_name, cfg.env.max_episode_steps,
                     resize_factor=cfg.env.resize_factor,
                     terminate_on_timeout=False,
                     thin=cfg.env.thin, specific_env=rand_env)
set_env_seed(env, cfg.seed + 1)


## Load eval environment
print("Loading Eval Env")
eval_env = env_load_fn(cfg.env.env_name, cfg.env.max_episode_steps,
                       resize_factor=cfg.env.resize_factor,
                       terminate_on_timeout=True,
                       thin=cfg.env.thin, specific_env=rand_env)
set_env_seed(eval_env, cfg.seed + 2)


## Set the observation space
obs_dim = env.observation_space['observation'].shape[0]
goal_dim = obs_dim
state_dim = obs_dim + goal_dim
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print(f'obs dim: {obs_dim}, goal dim: {goal_dim}, state dim: {state_dim}, action dim: {action_dim}, max action: {max_action}')


## Create the agent
from pud.ddpg import UVFDDPG
agent = UVFDDPG(
    state_dim, # concatenating obs and goal
    action_dim,
    max_action,
    **cfg.agent,
)
print(agent)


## Set up the Replay Buffer
from pud.buffer import ReplayBuffer
replay_buffer = ReplayBuffer(obs_dim, goal_dim, action_dim, **cfg.replay_buffer)


## Train the agent.
if train is not None:
    from pud.policies import GaussianPolicy
    policy = GaussianPolicy(agent)

    from pud.runner import train_eval, eval_pointenv_dists
    train_eval(policy,
                agent,
                replay_buffer,
                env,
                eval_env,
                eval_func=eval_pointenv_dists,
                **cfg.runner,
                )
    torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, 'agent.pth'))


## Get the trained model and put it in the agent.
ckpt_file = os.path.join(cfg.ckpt_dir, 'agent.pth')
agent.load_state_dict(torch.load(ckpt_file))
agent.eval()

## Implement the search policy
from pud.collector import Collector
env.set_sample_goal_args(prob_constraint=0.0, min_dist=0, max_dist=np.inf)
rb_vec = Collector.sample_initial_states(eval_env, replay_buffer.max_size)

## Get pairwise distances 
pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
from scipy.spatial import distance
euclidean_dists = distance.pdist(rb_vec)

## Visualize Graph
# from pud.visualize import visualize_graph
# visualize_graph(rb_vec, eval_env, pdist)

durationthing = 500
## Create Sparse graphical memory
from pud.policies import SparseSearchPolicy
search_policy = SparseSearchPolicy(agent, rb_vec, pdist=pdist, cache_pdist=True, max_search_steps=10)
eval_env.duration = durationthing

from pud.runner import cleanup_and_eval_search_policy
(initial_g, initial_rb, initial_success_rate, initial_eval_time), \
    (filtered_g, filtered_rb, filtered_success_rate, filtered_eval_time), \
    (cleaned_g, cleaned_rb, cleaned_success_rate, cleaned_eval_time, cleanup_time) = cleanup_and_eval_search_policy(search_policy, eval_env, 40, cleanup=int(1e5))


## Visualize initial and cleaned graphs for SGM
from pud.visualize import visualize_full_graph_name
visualize_full_graph_name(initial_g, initial_rb, eval_env, initial_success_rate, initial_eval_time, cfg, "init")
visualize_full_graph_name(filtered_g, filtered_rb, eval_env, filtered_success_rate, filtered_eval_time, cfg, "filtered")
visualize_full_graph_name(cleaned_g, cleaned_rb, eval_env, cleaned_success_rate, cleaned_eval_time, cfg, "cleaned")



# ## Create SoRB
from pud.policies import SearchPolicy
search_policy = SearchPolicy(agent, rb_vec, pdist=pdist, open_loop=True)
eval_env.duration = durationthing # We'll give the agent lots of time to try to find the goal.

## Visualize initial and cleaned graphs for SoRB
from pud.runner import cleanup_and_eval_search_policy_sorb
(initial_g, initial_rb, initial_success_rate, initial_eval_time), \
    (filtered_g, filtered_rb, filtered_success_rate, filtered_eval_time) = cleanup_and_eval_search_policy_sorb(search_policy, eval_env, 40)

## Visualize initial and cleaned graphs
from pud.visualize import visualize_full_graph_name
visualize_full_graph_name(initial_g, initial_rb, eval_env, initial_success_rate, initial_eval_time, cfg, "notsparseinit")
visualize_full_graph_name(filtered_g, filtered_rb, eval_env, filtered_success_rate, filtered_eval_time, cfg, "notsparsefiltered")