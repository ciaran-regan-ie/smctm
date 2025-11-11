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
uv run main.py task=<TASK> model=<MODEL> model.plastic=<PLASTIC>
```

where:
- `TASK` is one of: `CIFARFewShot`
- `MODEL` is one of: `LSTM`, `CTM`
- `PLASTIC` is one of: `true`, `false`