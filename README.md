# SMCTM: Self-Modifying Continuous Thought Machines

A self-modifying continuous thought machine.

## Experiments

### CIFAR Few Shot
Download the data from [google drive](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view):
```bash
uv run gdown 1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS -O data/raw_data/
unzip data/raw_data/CIFAR-FS.zip -d data/cifar_fs/
```

Run the train script with
```bash
uv run main.py task=CIFARFewShot model=CTM wandb.enabled=true
```