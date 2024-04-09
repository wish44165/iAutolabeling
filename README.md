## Iterative Refinement Strategy for Automated Data Labeling


The code has been successfully tested on Ubuntu 22.04.


<img src="https://github.com/wish44165/iAutolabeling/blob/main/assets/Fig2.png" alt="iAutolabeling" width="70%" >


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
$ for i in `seq 0 9`; do python main.py --curr_iter ${i} | tee iterLog${i}.txt; done
```

</details>


<details><summary>Folder Structure</summary>

### Initial

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
        └── main.py
```

### After executed

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
        ├── main.py
        ├── facial_v1.yaml, facial_v2.yaml, ...
        ├── iterLog0.txt, iterLog1.txt, ...
        └── runs/
            └── facial/
                ├── train/, train2/, ...
                └── predict/, predict2/, predict3/, predict4/, ...
```

</details>




### Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Model Prediction with Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/)
- [Problem](https://github.com/ultralytics/ultralytics/issues/1713#issuecomment-1605689756) ([solution](https://github.com/ultralytics/ultralytics/issues/2930#issuecomment-1571399356))




### Citation
```
@misc{chen2024iterative,
      title={Iterative Refinement Strategy for Automated Data Labeling: Facial Landmark Diagnosis in Medical Imaging}, 
      author={Yu-Hsi Chen},
      year={2024},
      eprint={2404.05348},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
