from causal.config import get_config
from causal.causal_controller.config import get_config as get_cc_config
from causal.causal_controller.CausalController import CausalController
from causal.causal_graph import get_causal_graph
import tensorflow as tf

config, _ = get_config()
cc_config, _ = get_cc_config()
cc_config.graph=get_causal_graph(config.causal_model)
cc_config.model_dir = ''
cc = CausalController(8, cc_config)

sess = tf.Session()
cc_config.pt_load_path = 'causal/logs/celebA_0319_124207/controller/checkpoints/CC-Model-2001'
cc.load(sess, cc_config.pt_load_path)
# fetch_fixed_z = {n.z:n.z for n in cc.nodes}
# feed_fixed_z = sess.run(fetch_fixed_z)

print('Check tvd after restore')
info=crosstab(self,report_tvd=True)
print('tvd after load:',info['tvd'])

cc.sample_label(sess, do_dict={'Mustache':1}, N=22)

from causal.main import get_trainer()




def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)



facade 
molding
cornice
pillar
window
door
sill
blind
balcony
shop
deco
background


path = glob('./datasets/%s/%s/*' % ('facades', 'train'))



def get_labels(path, img_idx):
    num_of_color = 12
    from matplotlib import cm
    color_map = [
        [0,0,221], # Background
        [1,48,251], # Facade
        [5,124,249], # Window
        [30,230,237], # Cornice
        [34,57,178], # Deco
        [110,53,122], # Balcony
        [121,233,161], # Sill
        [187,3,10], # Door
        [195,102,81], # Shop
        [195,249,72], # Pillar
        [249,81,5], # Molding
        [250,199,10] # Blind
    ]
    print(path[img_idx])
    img = imageio.imread(path[img_idx])
    h, w, _ = img.shape
    _w = int(w/2)
    img_A, img_B = img[:, :_w, :], img[:, _w:, :]
    # ds = []
    labels = [0]*12
    X = []
    for i in range(0, h, 3):
        for j in range(0, h, 3):
            X.append(img_B[i,j,:])
            max_dist = 9999999999999
            label = 0
            for k in range(len(color_map)):
                pix = img_B[i,j,:]
                dist = sum(abs(color_map[k] - pix))
                if dist < max_dist:
                    label = k
                    max_dist = dist
            labels[label] += 1
            # ds.append(max_dist)
    return np.array(labels) * (np.array(labels) > 50)

def generate_labels(path, output_path):
    with open(output_path, "w") as out:
        out.write("background facade window cornice deco balcony sill door shop pillar molding blind\n")
        for idx in range(len(path)):
            labels = get_labels(path, idx)
            labels_str = []
            for v in labels:
                if v > 0:
                    labels_str.append(str(1))
                else:
                    labels_str.append(str(-1))
            file_name = path[idx].split('/')[-1]
            out.write(file_name + " " + " ".join(labels_str) + "\n")



import scipy
import pandas as pd
from scipy import misc
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
import imageio


def get_labels(idx, num_c):
    X = []
    for img_idx in range(idx):
        img = scipy.misc.imread(path[img_idx], mode='RGB').astype(np.int32)
        h, w, _ = img.shape
        _w = int(w/2)
        img_A, img_B = img[:, :_w, :], img[:, _w:, :]
        
        for i in range(h):
            for j in range(h):
                X.append(img_B[i,j,:])
    kmeans = KMeans(n_clusters=num_c, random_state=0, n_jobs=4, verbose=1).fit(X)
    return pd.DataFrame(kmeans.labels_), pd.DataFrame(kmeans.cluster_centers_) 


color_map = [[0,0,170],
    [0,0,170],
    [0,85,256],
    [0,170,256],
    [0,256,256],
    [85,256,256],
    [170,256,170],
    [256,256,85],
    [256,170,0],
    [256,85,0],
    [256,0,0],
    [170,0,0]]



def get_label_dict(batch_size):
    with tf.Graph().as_default():
        trainer = get_trainer()
        cc = trainer.cc
        with trainer.sess as sess:
            return cc.sample_label(sess, do_dict={'sill':1}, N=batch_size)

attributes = pd.read_csv(config.attr_file,delim_whitespace=True) #+-1
