from model import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import pdb
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)


def main():
    ### create model
    model = Model()
    model.create_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()
