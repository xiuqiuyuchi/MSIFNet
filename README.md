# project introduction
This project is a remote sensing image classification model built on PyTorch

# Environmental requirements
matplotlib==3.5.1

mmcv==2.2.0

python==3.9

tensorboard==2.10.0

tensorflow==2.16.1

timm==0.9.12

torch==2.2.1+cu121

torchaudio==2.2.1+cu121

torchvision==0.17.1+cu121

tqdm==4.66.1

scikit-learn==1.4.0

# instructions
Dataset preparation

The model uses publicly available datasets of AID, UCM, and NWPU.

The AID, UCM, and NWPU datasets can be obtained from https://captain-whu.github.io/AID/ ,  http://weegee.vision.ucmerced.edu/datasets/landuse.html ,  http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html obtain.

training model

Firstly, change the root path in the get_data function using data.by, and regenerate class_indices.json, train_data.json, and test_data.json. Then, run them using train.by.
