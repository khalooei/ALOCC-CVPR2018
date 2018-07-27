import os
import numpy as np
from models import ALOCC_Model
from utils import pp, visualize, to_json, show_all_variables
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 40, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 1, "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size",128, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 45, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 45, "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string("dataset_address", "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

def main(_):
    """
    The main function for training steps     
    """
    pp.pprint(flags.FLAGS.__flags)
    n_per_itr_print_results = 100
    kb_work_on_patch = True

    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    # Manual Switchs ------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    # DATASET PARAMETER : UCSD
    #FLAGS.dataset = 'UCSD'
    #FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'

    nd_input_frame_size = (240, 360)
    nd_slice_size = (45, 45)
    n_stride = 25
    n_fetch_data = 600
    # ---------------------------------------------------------------------------------------------
    # # DATASET PARAMETER : MNIST
    # FLAGS.dataset = 'mnist'
    # FLAGS.dataset_address = './dataset/mnist'
    # nd_input_frame_size = (28, 28)
    # nd_slice_size = (28, 28)

    FLAGS.train = True

    FLAGS.input_width = nd_slice_size[0]
    FLAGS.input_height = nd_slice_size[1]
    FLAGS.output_width = nd_slice_size[0]
    FLAGS.output_height = nd_slice_size[1]

    FLAGS.sample_dir = 'export/'+FLAGS.dataset +'_%d.%d'%(nd_slice_size[0],nd_slice_size[1])
    FLAGS.input_fname_pattern = '*'

    check_some_assertions()

    # manual handling of GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        tmp_model = ALOCC_Model(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.batch_size,
                    attention_label=FLAGS.attention_label,
                    r_alpha=FLAGS.r_alpha,
                    dataset_name=FLAGS.dataset,
                    dataset_address=FLAGS.dataset_address,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    is_training = FLAGS.train,
                    log_dir=FLAGS.log_dir,
                    sample_dir=FLAGS.sample_dir,
                    nd_patch_size=nd_slice_size,
                    n_stride=n_stride,
                    n_per_itr_print_results=n_per_itr_print_results,
                    kb_work_on_patch=kb_work_on_patch,
                    nd_input_frame_size = nd_input_frame_size,
                    n_fetch_data=n_fetch_data)

        #show_all_variables()

        if FLAGS.train:
            print('Program is on Train Mode')
            tmp_model.train(FLAGS)
        else:
            if not tmp_model.load(FLAGS.checkpoint_dir)[0]:
                print('Program is on Test Mode')
                raise Exception("[!] Train a model first, then run test mode from file test.py")

if __name__ == '__main__':
  tf.app.run()
