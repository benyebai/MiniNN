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

````bash
git clone https://github.com/benyebai/MiniNN.git
cd MiniNN

uv sync
source .venv/bin/activate

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
````

Verify the installation:

```bash
python -c "from tensorgrad import Tensor; print(Tensor([1, 2, 3]))"
python -m unittest discover -s tests
```

## License

MIT
