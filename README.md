# SMCTM: Self-Modifying Continuous Thought Machines

A self-modifying continuous thought machine.

## Experiments

### Few-Shot Image Classification

The few-shot image classification experiments support two datasets: Few-Shot CIFAR-100 and MiniImageNet.

#### Dataset Setup

**CIFAR-FS (CIFAR-100 Few-Shot)**

Download the data from [Google Drive](https://drive.google.com/file/d/12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D/view):
```bash
uv run gdown 12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D -O data/raw_data/
unzip data/raw_data/CIFAR-FS.zip -d data/cifar_fs/
```

**MiniImageNet**

Download the data from [Google Drive](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view):
```bash
uv run gdown 1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS -O data/raw_data/
unzip data/raw_data/miniImageNet.zip -d data/miniImageNet/
```

#### Running Experiments

Run the training script:
```bash
uv run main.py task=<TASK> model=<MODEL> model.plastic=<PLASTIC>
```

**Parameters:**
- `TASK`: Choose from `FewShotCIFAR` or `FewShotMiniImageNet`
- `MODEL`: Choose from `LSTM` or `CTM`
- `PLASTIC`: Set to `true` for plastic models or `false` for non-plastic models

**Examples:**
```bash
# Train CTM on CIFAR-FS with plasticity
uv run main.py task=FewShotCIFAR model=CTM model.plastic=true

# Train LSTM on MiniImageNet without plasticity
uv run main.py task=FewShotMiniImageNet model=LSTM model.plastic=false
```

