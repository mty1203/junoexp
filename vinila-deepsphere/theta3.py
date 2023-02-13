import os
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer

font = {'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2","/gpu:3","/gpu:4","/gpu:5","/gpu:6","/gpu:7"])

class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.98
    self.model.optimizer.lr.assign(new_lr)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

def draw_performance(x,y,title):

  plt.clf()
  plt.figure(figsize=(12,8))
  plt.grid()
  plt.scatter(x,y, label="predictions", color='red')
  mi = np.amin(x)
  ma = np.amax(x)
  plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x")
  plt.legend()
  plt.xlabel("True")
  plt.ylabel("Predicted")
  plt.title(title)
  plt.savefig("./result4/%s.png" %title)
def normalization(feature):
  norms=np.linalg.norm(feature,axis=1)
  return feature/norms[:,None]

def draw_hists(y_true,y_predict):
  size = y_true.shape[0]
  m = {}
  for i in range(size):
    true_theta = y_true[i]
    predict_theta = y_predict[i]
    if true_theta in m:
      m[true_theta].append(predict_theta)
    else:
      m[true_theta] = [predict_theta]

  for k,v in m.items():
    plt.clf()
    plt.figure(figsize=(12,8))
    plt.hist(v, bins=20,color='red',histtype='stepfilled',alpha=0.75)
    plt.title("True theta=%f" % k)
    plt.savefig("theta_%f.png" % k)

if __name__ == "__main__":

  nside = 32
  npix = hp.nside2npix(nside)
  print("n pixel: %d" % npix)
  indices = np.arange(hp.nside2npix(nside))

  #x_fht = np.load("/home/duyang/j/x_fht.npy")
  #x_npe = np.load("/home/duyang/j/x_npe.npy")
  #x_slope = np.load("/home/duyang/j/x_slope.npy")

  x_fht = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_fht.npy")
  
  x_npe = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_npe.npy")
  x_slope = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_slope.npy")
  #x_nperatio = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_nperatio.npy")
  x_npe_s = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/x_npe_s.npy")
  x_fht_s = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/x_fht_s.npy")
  #x_peak = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_peak.npy")
  #x_peaktime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_peaktime.npy")
  x_npe=normalization(x_npe)
  x_slope=normalization(x_slope)
  x_peak=normalization(x_npe_s)

  #x_mediantime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_mediantime.npy")
  #x_meantime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_meantime.npy")
  #x_rmstime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_rmstime.npy")
  #x_skewtime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_skewtime.npy")
  #x_kurtime = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_kurtime.npy")
  #x_fht_s = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_fht_s.npy")
  #x_npe_s = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/x_npe_s.npy")
  #x_nperatio = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_NPERATIO.csv", dtype=np.float,delimiter=',')
  #x_peak = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_PEAK.csv", dtype=np.float,delimiter=',')
  #x_peaktime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_PEAKTIME.csv", dtype=np.float,delimiter=',')
  #x_mediantime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_MEDIANTIME.csv", dtype=np.float,delimiter=',')
  #x_meantime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_MEANTIME.csv", dtype=np.float,delimiter=',')
  #x_rmstime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_RMSTIME.csv", dtype=np.float,delimiter=',')
  #x_skewtime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_SKEWTIME.csv", dtype=np.float,delimiter=',')
  #x_kurtime = np.loadtxt("/home/liteng/work/JUNO/dataset_nu_elec_all_corr/ElecNu_KURTIME.csv", dtype=np.float,delimiter=',')

  x_fht[x_fht==1250]=0
  x_fht=normalization(x_fht)
  x_fht_s[x_fht_s==1250]=0
  x_peaktime=normalization(x_fht_s)
  x_all = np.stack((x_fht,x_npe,x_slope,x_peak,x_peaktime),axis=-1)
  y_all = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/y_all.npy")[:,0]
  y_enu = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/y_all.npy")[:,6]
  y_emu = np.load("/home/liteng/work/JUNO/dataset_npy/det_32/sep/y_all.npy")[:,7]

  #x_fht_HE = x_fht[y_enu > 2]
  #x_npe_HE = x_npe[y_enu > 2]
  #x_slope_HE = x_slope[y_enu > 2]
  x_HE = x_all[y_enu > 1.0]
  y_HE = y_all[y_enu > 1.0]
  y_enu_HE = y_enu[y_enu > 1.0]

  #x_sel = x_HE[y_enu_HE<2.3]
  #y_sel = y_HE[y_enu_HE<2.3]
  #y_all = np.load("/home/duyang/j/y_all.npy")

  x_train, x_test, y_train, y_test = train_test_split(x_HE, y_HE, test_size=0.10, random_state=20)

  print(len(x_train),len(y_train))
  '''
  layers = [hp_layer.HealpyChebyshev(K=10, Fout=40, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=40, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, Fout=20, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=20, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, Fout=10, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=10, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, Fout=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)]
  '''
  layers = [hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
            #hp_layer.HealpyPool(p=1),
            hp_layer.HealpyPseudoConv(p=1, Fout=24, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=36, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, Fout=36, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
            #hp_layer.HealpyPool(p=1),
            hp_layer.HealpyPseudoConv(p=1, Fout=24, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=6, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
           # hp_layer.HealpyPseudoConv(p=1, Fout=10, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)]

  tf.keras.backend.clear_session()

  with strategy.scope():
    model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=40)
    batch_size = 128
    model.build(input_shape=(None, len(indices), 5))
    model.summary(110)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0004),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()]
    )

  if os.path.exists('checkpoint'):
    model.load_weights('./result4/weights_5features_sPMT.h5')

  else:
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[LearningRateReducerCb(),es_callback]
    )

    model.save_weights('./result4/weights_5features_sPMT.h5')

    plt.figure(figsize=(12,8))
    plt.plot(history.history["loss"], label="training")
    plt.plot(history.history["val_loss"], label="validation")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.savefig("learning_curve_theta.png")

  print(model.evaluate(x_train, y_train))
  print(model.evaluate(x_test, y_test))

  predictions = model.predict(x_test)
  draw_performance(y_test, predictions, 'prediction_theta')

  #draw_hists(y_test, predictions)
  np.save("./result4/theta_predict.npy", predictions)
  np.save("./result4/theta_true.npy", y_test)

  predictions = model.predict(x_train)
  draw_performance(y_train, predictions, 'prediction_theta_train')
  np.save("./result4/theta_predict_train.npy", predictions)
  np.save("./result4/theta_true_train.npy", y_train)
