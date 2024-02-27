# iterative auto labeling

iterative auto label strategy combine with nms


<details><summary>Create Conda Environment</summary>

```bash
$ conda create -n yolov8 python=3.10 -y
$ conda activate yolov8
$ https://github.com/wish44165/iAutolabeling
$ cd iAutolabeling/
$ pip install ultralytics
```

</details>


<details><summary>Commands</summary>

```bash
# demo
$ for i in `seq 0 4`; do python train_iterative.py --n_epoch 4 --curr_iter ${i}; done

# main
$ for i in `seq 0 9`; do python train_iterative.py --curr_iter ${i}; done
```

</details>


<details><summary>Folder Structure</summary>

## Initial folder structure

```bash
ICME2024/
├── datasets/
    └── v0/
        ├── images/
            ├── train/
            └── val/
        └── labels/
            ├── train/
            └── val/
└── src/
    └── iAutolabeling/
        ├── facial.yaml
        └── iAutolabel.py
```

## After executed folder structure

```bash
ICME2024/
├── datasets/
    ├── v0/
        ├── images/
            ├── train/
            └── val/
        └── labels/
            ├── train/
            └── val/
    └── v1/, v2/, ...
├── src/
    └── iAutolabeling/
        ├── facial.yaml
        ├── iAutolabel.py
        └── runs/
            └── facial/
                ├── train/, train2/, ...
                └── predict/, predict2/, predict3/, predict4/, ...
```

</details>


## Acknowledgments and References

- [Ultralytics](https://github.com/ultralytics/ultralytics)
