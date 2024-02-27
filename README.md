# iterativeLabel
iterative label strategy combine with nms


```bash
# demo
$ for i in `seq 0 4`; do python train_iterative.py --n_epoch 4 --curr_iter ${i}; done


$ for i in `seq 0 9`; do python train_iterative.py --curr_iter ${i}; done
```
