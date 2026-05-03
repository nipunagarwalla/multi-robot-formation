## Repository layout

```
.
├── train.py          # PPO training loop
├── model.py          # GNN
├── env_line.py       # Line formation env
├── env_pentagon.py   # Circle formation env
├── env_wedge.py      # Wedge formation env
├── weights/          # Trained checkpoints are saved here
└── README.md
```

## Requirements

- Python 3.8+
- `torch`
- `torch_geometric`
- `torch_cluster`
- `numpy`
- `gymnasium`
- `pygame`
- `scipy`
- `tqdm`

Install (example, adjust for your CUDA/CPU build):

```bash
pip install numpy scipy tqdm gymnasium pygame
pip install torch
pip install torch_geometric torch_cluster
```

`torch_cluster` usually needs to match your installed torch version; see the PyG install docs for the correct wheel URL for your setup.

## How to run training

From the project root:

```bash
python train.py
```

The script:
- Uses `cpu` by default (see `device = 'cpu'` in `train.py`; change to `'cuda'` to train on GPU).
- Sets `SDL_VIDEODRIVER=dummy` so pygame can initialize without a display (useful on headless machines).
- Prints `mean rewards at iter N: …` after each rollout and `loss: …` after each PPO update block.
- Saves checkpoints to `weights/real-line2/weights_epoch{iteration}.pt` (the directory is created automatically).

### Switching formations

`train.py` imports from `env_line` by default:

```python
from env_line import PassageEnv
```

To train the pentagon or wedge variant, change that single import to:

```python
from env_pentagon import PassageEnv
# or
from env_wedge import PassageEnv
```

and update `agent_formation` in `train.py` to match the chosen shape (the line / pentagon / wedge options are already listed in the file — comment/uncomment as needed).

### Resuming / using a checkpoint

Checkpoints saved by `train.py` are plain `torch.save(agent.state_dict(), …)` files, so they can be loaded into a freshly-built `Agent` with:

```python
agent.load_state_dict(torch.load("weights/real-line2/weights_epoch{N}.pt"))
```

before resuming rollouts or running evaluation.
