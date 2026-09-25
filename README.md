# MiniNN

Project 3/10 in a from-first-principles Dreamer build.

MiniNN is a small neural-network library built on
[TensorGrad](https://github.com/benyebai/TensorGrad). The project develops
tensor parameters, modules, linear layers, activations, losses, initialization,
Adam, gradient clipping, and checkpointing without using a machine-learning
framework.

## Requirements

- Python 3.10 or newer
- Git, because TensorGrad is installed directly from its GitHub repository

## Setup

```bash
git clone https://github.com/benyebai/MiniNN.git
cd MiniNN
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Verify the installation:

```bash
python -c "from tensorgrad import Tensor; print(Tensor([1, 2, 3]))"
python -m unittest discover -s tests
```

If your editor reports that `mininn` cannot be resolved, select
`.venv/bin/python` as the workspace interpreter and restart its language
server. The repository's basedpyright configuration uses that environment.

## Development roadmap

1. Parameter and module contracts
2. Linear layers and activations
3. Loss functions and initialization
4. Adam and gradient clipping
5. Checkpoint save/load
6. An MLP that reliably learns XOR

## License

MIT
