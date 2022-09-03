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

```
cd integrated_models/nat_disguiser
```

And run,

```
python net_disguiser.py --gpu [gpuid]
```

### build transbench

```
cd transbench
```

If you want to build transbench yourself,

```
cd high_fidelity
```

And run,

```
python build_high_fidelity_transbench.py
```

The already build transbench written in ```yaml``` is in ```data/high_fidelity```

### train predictor

```
cd predictors
```

If you want to train high fidelity predictor,

```
cd high_fidelity
```

```
python finetune_predictor.py
```

The config file of predictor and the config file of optimization detail is in the two yamls.

### train predictor based tansformer

The code is in 

```
integrated_models/predictor_based
```

### final test

The code is in 

```
final_test/compile_based
```

