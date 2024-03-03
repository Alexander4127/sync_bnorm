# SyncBatchNorm implementation

This project implements the synchronized batch normalization layer with forward/backward passes.
It is compared with torch implementation on several benchmarks

## Report

Solution details and experiments results are available in `report.ipynb`.

## Benchmarking

#### BatchNorm benchmark

Direct comparison of custom and library versions with time/memory benchmarking are executed with
```shell
python bn_benchmark.py --size=<number of GPU> --norm_type=<selected_implementation custom or lib>
```

It runs `forward/backward` methods with `hidden_size` in `[128, 256, 512, 1024]` and `batch_size` in `[32, 64]`.

#### Training benchmark

More complex full training pipelines supports more 
```shell
python training_benchmark.py \
    --norm_type=<selected_implementation custom or lib> \
    --size=<number of GPU> \
    --batch_size=<selected batch size for training> \
    --grad_accum=<grad accumulation steps> \
    --n_epoch=<number of training epochs> \
    --run_val=<run validation epoch or not>
    
```

## Tests

Correctness tests are provided in `test_syncbn.py` and can be run by
```shell
pytest .
```
