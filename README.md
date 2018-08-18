# Adversarially Learned One-Class Classifier for Novelty Detection (ALOCC-CVPR2018)
[[CVPR Poster]](cvpr18_poster_ALOCC.pdf) [[Project]](https://github.com/khalooei/ALOCC-CVPR2018) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf)

As you know, this work inspired by the success of generative adversarial networks for training deep models in unsupervised and semi-supervised settings,we propose an end-to-end architecture for one-class classification. Our architecture is composed of two deep networks, each of which trained by competing with each other while collaborating to understand the underlying concept in the target class, and then classify the testing samples. One network works as the novelty detector, while the other supports it by enhancing the inlier samples and distorting the outliers. The intuition is that the separability of the enhanced inliers and distorted outliers is much better than deciding on the original samples. 

Here is the preliminary version of the code on grayscale databases. 

Please feel free to contact me if you have any questions.

## overview
<p align="center"><img src='imgs/overview.jpg'></p>

**Note**: 
The current software works well with TensorFlow 1.2

## Prerequisites
- Windows, Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install TensorFlow and dependencies from https://tensorflow.org
```bash
# GPU version
pip install tensorflow-gpu

#CPU version
pip install tensorflow
```
- Install pillow,scipy,skimage,imageio,numpy,matplotlib
```
pip install numpy scipy scikit-image imageio matplotlib pillow
```
- Clone this repo:
```
git clone https://github.com/khalooei/ALOCC-CVPR2018
cd ALOCC-CVPR2018
```


<p align="center"><img src='imgs/architecture.jpg'></p>


### ALOCC train
- Download a UCSD dataset:
```
mkdir dataset
cd dataset
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
tar -xzf UCSD_Anomaly_Dataset.tar.gz
```

- Config path of dataset and dataset name :
```
# for train on UCSD and patch_size 45*45
python train.py --dataset UCSD --dataset_address ./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train --input_height 45 --output_height 45 

# for train on MNIST
python train.py --dataset mnist --dataset_address ./dataset/mnist/ --input_height 28 --output_height 28
```


## ALOCC's Cheat sheet
Some of our followers said that if ALOCC have any cheat-sheet, it would have more easy for readers and followers do in their own research or industries. (Thanks a lot Andy Steinbach and other followers for mentioning this note item)
* At first, You must prepare your dataset if you want to use on our work. Therefore, you can look at the code as mentioned like UCSD or MNSIT dataset. If your own dataset is like UCSD or MNSIT, we prefer to do the same step for loading or use better solutions to import as UCSD/MNIST dataset. Then you can name your data set as UCSD/MNIST (for e.g. MyDataSetName as dataset name). It means that you can add loading code of your dataset in `models.py` file from line 180 up to 190 as UCSD/MNIST.
In this time, you can pass a parameter as bellow :
```
--dataset MyDataSetName --dataset_address ./pathOfMyDataSetName
```
* You must set your patch_size. It means that your image wants to be sliced into patch_size in array (for e.g. (60,60)) as bellow:
```
--input_height 60 --output_height 60
```
* To view more parameter of this pseudo implementation, you could follow the 'train.py' file
* To change some structure or architecture of ALOCC, you can change in `model.py` file. We have some pre/post processing functions in `kh_tools.py` and  `utils.py`. Also we have some abstract function of Conv or any similar functions in `ops.py`. You can change the test process in `test.py` and also `model.py` from line 460 up to the end.
* If you have any further questions, please don't hesitate to contact me :rose: .
* Also, contributions are welcome. If you implement our paper in other framework or implementation style, you can contact me to publish your link on this page.


<hr>
- To view training results you can access them from export directory

### ALOCC test
- You can run the following commands:
```
# for test on UCSD and patch_size 45*45 and some specific dir like ['Test004'], etc. We prefer to open test.py file and edit every thing that you want
python test.py --dataset UCSD --dataset_address ./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train --input_height 45 --output_height 45 

# for test on MNIST
python test.py --dataset mnist --dataset_address ./dataset/mnist/ --input_height 28 --output_height 28
```

<hr>

### Apply a pre-trained model (ALOCC)
- You can use a pretrained model and locate directory path to train.py or test.py file. 
- The pretrained model is saved at 
```
./checkpoints/{datasetname}_{batch_size}_{patch_size}
```
- To view test results you can access them from samples directory

<hr>

#### MNIST
input:
<br>
<img src='imgs/MNIST__ALOCC_input.jpg' align="center" width=384>
<br>
Generated:
<br>
<img src='imgs/MNIST__ALOCC_generated.jpg' align="center" width=384>


#### UCSD
input:
<br>
<img src='imgs/UCSD__ALOCC_input.jpg' align="center" width=384>
<br>
Generated:
<br>
<img src='imgs/UCSD__ALOCC_generated.jpg' align="center" width=384>


<hr>

## Citation
If you use this code or idea for your research, please cite our papers.

```
@inproceedings{sabokrou2018adversarially,
  title={Adversarially Learned One-Class Classifier for Novelty Detection},
  author={Sabokrou, Mohammad and Khalooei, Mohammad and Fathy, Mahmood and Adeli, Ehsan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3379--3388},
  year={2018}
}
```
