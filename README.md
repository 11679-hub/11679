#### This is the demo code of our paper "Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising" in submission to CVPR 2025.

This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained model.
- README file.

This repo can reproduce the main results in Table (1) of our main paper.
All the source code and pre-trained models will be released to the public for further research.


#### 1. Create Environment:

------

- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch == 2.0.0](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```


#### 2. Prepare Dataset:


To use the DVD dataset, please follow the steps below:

1. Download the  Test Dataset:
   Download the test dataset from [DVN GitHub Repository](https://github.com/megvii-research/DVN).

2. Organize the Dataset:
  Then, unzip the file into `code/Dataset` directory.
  And the directory structure is organized as:
  
  ```
  Dataset
  ├── DVD_test
  │     ├── RGB
  │     ├── NIR
  ```

#### 3. Testing

3. 1 Test our pre-trained FCENet models on the DVD dataset with different noise levels (2,4,6). The results will be saved in 'code/results'.

```shell
python test.py --gpu_id=0 --sigma=2

python test.py --gpu_id=0 --sigma=4

python test.py --gpu_id=0 --sigma=6

```

#### 4. This repo is mainly based on DVN.  In our experiments, we use the following repos:
DVN: https://github.com/megvii-research/DVN

We extend our sincere appreciation and gratitude for the valuable contributions made by these repositories.
