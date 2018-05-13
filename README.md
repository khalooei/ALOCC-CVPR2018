# ALOCC-CVPR2018
Adversarially Learned One-Class Classifier for Novelty Detection  [[Presentation file]](#) [[TensorFlow Code]](#) [[Pytorch Code]](#) [[Paper]](https://arxiv.org/pdf/1802.09088.pdf)

As you know, this work inspired by the success of generative adversarial networks for training deep models in unsupervised and semi-supervised settings,we propose an end-to-end architecture for one-class classification. Our architecture is composed of two deep networks, each of which trained by competing with each other while collaborating to understand the underlying concept in the target class, and then classify the testing samples. One network works as the novelty detector, while the other supports it by enhancing the inlier samples and distorting the outliers. The intuition is that the separability of the enhanced inliers and distorted outliers is much better than deciding on the original samples. 

We are busy now and try to publish the clean code as soon as possible.
Please feel free to contact me if you have any questions.

If you find this idea useful in your research, please cite:
```
@article{DBLP:journals/corr/abs-1802-09088,
  author    = {Mohammad Sabokrou and
               Mohammad Khalooei and
               Mahmood Fathy and
               Ehsan Adeli},
  title     = {Adversarially Learned One-Class Classifier for Novelty Detection},
  journal   = {CoRR},
  volume    = {abs/1802.09088},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.09088},
  archivePrefix = {arXiv},
  eprint    = {1802.09088},
  timestamp = {Fri, 02 Mar 2018 13:46:22 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1802-09088},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
