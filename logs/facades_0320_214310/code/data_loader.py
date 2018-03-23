import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from causal.main import get_trainer
import tensorflow as tf
import pandas as pd 

class DataLoader():

    def __init__(self, dataset_name, img_res=(128, 128), batch_size=1):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.attributes = pd.read_csv('data/facades/train_labels.txt',delim_whitespace=True)
        self.pos_w_pos_c_img_names = list(self.attributes[(self.attributes['window'] > 0) & (self.attributes['cornice'] > 0)].index)
        self.pos_w_neg_c_img_names = list(self.attributes[(self.attributes['window'] > 0) & (self.attributes['cornice'] < 0)].index)
        self.neg_w_pos_c_img_names = list(self.attributes[(self.attributes['window'] < 0) & (self.attributes['cornice'] > 0)].index)
        self.neg_w_neg_c_img_names = list(self.attributes[(self.attributes['window'] < 0) & (self.attributes['cornice'] < 0)].index)
        self.label_dicts = []
        self.count = 0
        self.get_label_dict(batch_size)


    def get_label_dict(self, batch_size):
        with tf.Graph().as_default():
            self.trainer = get_trainer()
            self.cc = self.trainer.cc
            with self.trainer.sess as sess:
                print("\nInit causal labels...\n")
                for i in range(10000):
                    self.label_dicts.append(self.cc.sample_label(sess, N=batch_size))
    
    def get_image_paths(batch_size):
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))


    def load_data(self, batch_size=1, is_testing=False):
        label_dict = self.label_dicts[self.count]
        self.count += 1
        print("Causal labels:", label_dict)

        pos_w_pos_c = sum((label_dict['window'] > 0.5) & (label_dict['cornice'] > 0.5))
        pos_w_neg_c = sum((label_dict['window'] > 0.5) & (label_dict['cornice'] <= 0.5))
        neg_w_pos_c = sum((label_dict['window'] <= 0.5) & (label_dict['cornice'] > 0.5))
        neg_w_neg_c = sum((label_dict['window'] <= 0.5) & (label_dict['cornice'] <= 0.5))

        data_type = "train" if not is_testing else "test"

        if is_testing:
            path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
            batch_images = np.random.choice(path, size=batch_size)
            print ("Sample image:", batch_images)
        else:
            data_path = './datasets/%s/%s/' % (self.dataset_name, data_type)
            pos_w_pos_c_images_names = np.random.choice(self.pos_w_pos_c_img_names, size=pos_w_pos_c)
            pos_w_neg_c_images_names = np.random.choice(self.pos_w_neg_c_img_names, size=pos_w_neg_c)
            neg_w_pos_c_images_names = np.random.choice(self.neg_w_pos_c_img_names, size=neg_w_pos_c)
            neg_w_neg_c_images_names = np.random.choice(self.neg_w_neg_c_img_names, size=neg_w_neg_c)
            batch_images_names = np.concatenate([pos_w_pos_c_images_names, 
                                                 pos_w_neg_c_images_names,
                                                 neg_w_pos_c_images_names,
                                                 neg_w_neg_c_images_names])
            batch_images = [data_path + n for n in batch_images_names]
            print ("Sample image:", pos_w_pos_c_images_names, 
                                                 pos_w_neg_c_images_names,
                                                 neg_w_pos_c_images_names,
                                                 neg_w_neg_c_images_names )
       

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
