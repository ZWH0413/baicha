# 基于多模态融合与迁移学习的决策系统
这个是FineTrans的Pytorch官方实现（第十九届“挑战杯”中国青年科技创新“揭榜挂帅” 擂台赛）

> **基于跨模态知识迁移与表征对齐的残差特征蒸馏**
>
> 

## 环境安装
```bash
# create environment
conda create -n FineTrans python=3.9.12

# install pytorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install clip
pip install opencv-clip

# install fvcore
pip install 'git+https://github.com/facebookresearch/fvcore'

# install pytorchvideo
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```

## 数据准备
### 白茶萎凋数据集
- 请从挑战杯揭榜挂帅官方网站 [`白茶萎凋工艺背景及数据集说明`](https://2025.tiaozhanbei.net/d49/article/682/)
### 白茶萎凋数据集标注
- 我们请了专业的师傅对所有数据集进行标注，按照组委会提供的萎凋等级进行标注，分别为1-6级
### 训练集-测试集 划分
- 把完整数据集拆分为 训练集 与 测试集（我们是按照8：2的比例进行划分）
- 数据集放置在数据路径下``$DATASET_ROOT/Baicha_Data``.
- 把config文件中的训练与测试数据地址``DATALOADER.TRAIN.IMAGE_PATH``, ``DATALOADER.TEST.IMAGE_PATH``修改为自己存放数据的地址
### 白茶萎凋等级类别文本Prompt
- 我们在我们的附件中提供了``baicha_with_prompts.json``，请在config文件中把``MODEL.TEXT_PROMPT_DICT``修改为文本Prompt的地址

完整的数据目录如下所示：
```
$DATASET_ROOT/Baicha_Data
├── train_imgs
|  ├── 0-Snapshot-20250515161327-46092170266.JPG
|  ...
|  └── 5-Snapshot-20250518101453-2422929661196.JPG
|  ...
├── test_imgs
|  ├── 0-Snapshot-20250515161326-46089464764.JPG
|  ...
|  └── 5-Snapshot-20250518101453-2422928511770.JPG
|  ...
├── annotations
|  ├── baicha_text_prompts.json
```

## 预训练模型
### CLIP
- config文件中``MODEL.ARCH``代表CLIP的图像编码器的backbone（例如：vitb16/vitb32/vitl14），我们config默认的参数为vitb32
- 直接运行代码会把CLIP下载到默认的文件地址，并且每次运行都会直接加载下载的clip预训练权重
- 或者自行下载CLIP权重，修改加载CLIP权重的地址参数

## Model Zoo
我们附件中提供了我们训练好的模型，如果直接推理请修改config文件中的 ``TEST.MODEL_WEIGHTS``.

## 训练和测试
### 在白茶萎凋数据集上进行训练

```bash
python train.py
```
### 在白茶萎凋数据集上进行测试

```bash
python test.py
```