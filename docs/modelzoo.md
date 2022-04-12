# Model Zoo

Opt: Optimizer during training <br>
Iter: Training iterations<br>
Aux: Apply auxiliary loss during training

## Pascal Context

| Methods | Backbone |Opt | Iter | Aux |SS mIOU(%) |MF mIOU(%) |ckpts | Logs|
|  ----   | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| FCN   | ResNet-50 (D8) |SGD|30K|| 47.72 |-|[Baidu Drive](https://pan.baidu.com/s/1z5YL-b08DRAvchYo5XLtug?pwd=7zvt)|[tb.dev](https://tensorboard.dev/experiment/oSdzE8OMTxO4KYerRjzolg/#scalars)|
| FCN + CAR | ResNet-50 (D8) |SGD|30K|| 48.40 |-|[Baidu Drive](https://pan.baidu.com/s/1lZm6fVY0EDWr8PB2aFzLUQ?pwd=wckt)|[tb.dev](https://tensorboard.dev/experiment/sL1TxrwvTnKCRWDEykDReA/#scalars)|
| Self-Attention   | ResNet-50 (D8) |SGD|30K|| 48.32 |-|[Baidu Drive](https://pan.baidu.com/s/1y9nii5fSB2QgJCjJNW5q2A?pwd=h97a)|[tb.dev](https://tensorboard.dev/experiment/UD542RNyRzy2REVYya4slQ/#scalars&_smoothingWeight=0)|
| Self-Attention + CAR| ResNet-50 (D8) |SGD|30K|| 50.50 |-|[Baidu Drive](https://pan.baidu.com/s/1ZvZ_70Qmp7ctM-IDbIy6Yw?pwd=matl)|[tb.dev](https://tensorboard.dev/experiment/PvnP9wNZTZeTEDnEbq6kaA/#scalars&_smoothingWeight=0)|
| CAR  | ConvNeXt-L (JPU) |SGD|30K|| 60.48 |61.80|[Baidu Drive](https://pan.baidu.com/s/1QKx-WjDo2FdSIjM_JAKFUw?pwd=px3z)|[tb.dev](https://tensorboard.dev/experiment/a5qQ7H66QCe40VQOmCv4QQ/#scalars&_smoothingWeight=0)|
| CAR + CAR | ConvNeXt-L (JPU) |SGD|30K|| 61.40 |62.69|[Baidu Drive](https://pan.baidu.com/s/1393t1xpcRS-_C3NeZVOMgw?pwd=1b8z)|[tb.dev](https://tensorboard.dev/experiment/sdo4vDgnR9KFZdaF1QN4EA/#scalars&_smoothingWeight=0)|
| CAA + CAR  | ConvNeXt-L (JPU) |Adam|40K|&check;| 62.97 |64.12|[Baidu Drive](https://pan.baidu.com/s/1vSITtQ5lIilQK-4IfayAiA?pwd=k9ja)|[tb.dev](https://tensorboard.dev/experiment/xEL5BOfETNKTE93bjgcwdg/#scalars&_smoothingWeight=0)|

More ckpts will be uploaded later (I am busy atm), create an issue if you need them now.
