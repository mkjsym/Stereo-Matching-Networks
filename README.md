This is Stereo Matching Network Based on PSMNet

## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
```

### Train
As an example, use the following command to train a PSMNet on Scene Flow

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

As another example, use the following command to finetune a PSMNet on KITTI 2015

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```
You can also see those examples in run.sh.

### Evaluation
Use the following command to evaluate the trained PSMNet on KITTI 2015 test data

```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

Update: 2018/9/6 We released the pre-trained KITTI 2012 model.

Update: 2021/9/22 a pretrained model using torch 1.8.1 (the previous model weight are trained torch 0.4.1)

| KITTI 2015 |  Scene Flow | KITTI 2012 | Scene Flow (torch 1.8.1)
|---|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view?usp=sharing)| [Google Drive](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view?usp=sharing)

### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results

※Note that the reported 3-px validation errors were calculated using KITTI's official matlab code, not our code.

### Visualization of Receptive Field
We visualize the receptive fields of different settings of PSMNet, full setting and baseline.

Full setting: dilated conv, SPP, stacked hourglass

Baseline: no dilated conv, no SPP, no stacked hourglass

The receptive fields were calculated for the pixel at image center, indicated by the red cross.
