# SMCTM: Self-Modifying Continuous Thought Machines

A self-modifying continuous thought machine.

## Experiments

### Few-Shot Image Classification

The few-shot image classication experiments use either the Few-Shot CIFAR100 dataset or the MiniImageNet dataset

#### Few-Shot CIFAR

Download the data from [google drive](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view):
```bash
uv run gdown 1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS -O data/raw_data/
unzip data/raw_data/miniImageNet.zip -d data/miniImageNet/
```

#### MiniImageNet
Download the data from [google drive](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view):
```bash
uv run gdown 12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D -O data/raw_data/
unzip data/raw_data/CIFAR-FS.zip -d data/cifar_fs/
```

Run the train script with
```bash
uv run main.py task=<TASK> model=<MODEL> model.plastic=<PLASTIC>
```

where:
- `TASK` is one of: `FewShotImageClassification`
- `MODEL` is one of: `LSTM`, `CTM`
- `PLASTIC` is one of: `true`, `false`

