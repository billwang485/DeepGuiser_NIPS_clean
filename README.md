# DeepGuiser Tutorial

## General Flow

### NAT-D Flow

1. Pretrain supernet
2. Train transformer by training integrated model

### Predictor Flow

1. Pretrain supernet
2. Build Transbench
3. Train Predictor on Transbench
4. Train transformer with predictor

## Detailed Tutorial

## pretrain_supernet

```bash
cd supernet/
```

To train **one** supernet, run

```python
python train_supernet.py --gpu [gpuid] --seed [seed_num]
```

After training complete, copy supernet and twin supernet to ``${workspaceFolder}/supernet/selected_supernet`` and rename them to ``supernet.pt`` and ``supernet_twin.pt``

### train gates integrated model



### build transbench

