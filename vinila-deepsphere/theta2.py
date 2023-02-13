import os
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tf_siren import SinusodialRepresentationDense
font = {'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])


class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.96
    self.model.optimizer.lr.assign(new_lr)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

def m_acos(x):
    return K.acos(x)

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
  plt.savefig("%s.png" %title)

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
def m_acos(x):
  return tf.acos(x)
if __name__ == "__main__":

  nside = 32
  npix = hp.nside2npix(nside)
  print("n pixel: %d" % npix)
  indices = np.arange(hp.nside2npix(nside))

  x_fht = np.load("x_fht.npy")
  x_fht[x_fht==1250]=-1
  x_npe = np.load("x_npe.npy")
  x_slope = np.load("x_slope.npy")
  x_nperatio = np.load("x_nperatio.npy")
  x_peak = np.load("x_peak.npy")
  x_peaktime = np.load("x_peaktime.npy")
  #x_mediantime = np.load("ElecNu_MEDIANTIME.npy")
  #x_meantime = np.loadtxt("/disk1/liteng/work/JUNO/dataset_nu_elec_all_32/ElecNu_MEANTIME.csv", dtype=np.float,delimiter=',')
  #x_rmstime = np.loadtxt("/disk1/liteng/work/JUNO/dataset_nu_elec_all_32/ElecNu_RMSTIME.csv", dtype=np.float,delimiter=',')
  #x_skewtime = np.loadtxt("/disk1/liteng/work/JUNO/dataset_nu_elec_all_32/ElecNu_SKEWTIME.csv", dtype=np.float,delimiter=',')
  #x_kurtime = np.loadtxt("/disk1/liteng/work/JUNO/dataset_nu_elec_all_32/ElecNu_KURTIME.csv", dtype=np.float,delimiter=',')
  #x_all = np.stack((x_fht,x_npe,x_slope,x_nperatio,x_peak,x_peaktime,x_mediantime,x_meantime,x_rmstime,x_skewtime,x_kurtime),axis=-1)
  x_all = np.stack((x_fht,x_npe,x_slope,x_nperatio,x_peak,x_peaktime),axis=-1)
  y_all = np.load("y_all.npy")[:,0]

  x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.20, random_state=42)
  #get_custom_objects().update({'m_acos': Activation(m_acos)})
  layers=[  hp_layer.HealpyChebyshev(K=10, Fout=10, use_bias=True, use_bn=True, activation="relu"), 
            hp_layer.HealpyChebyshev(K=10, Fout=10, use_bias=True, use_bn=True, activation=None),
            SinusodialRepresentationDense(10,activation='sine', w0=1.0),
            hp_layer.HealpyPseudoConv(p=1, Fout=10, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=10, use_bias=True, use_bn=True, activation="relu"), 
            hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation=None),
            SinusodialRepresentationDense(5,activation='sine', w0=1.0),
            hp_layer.HealpyPseudoConv(p=1, Fout=5, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPseudoConv(p=1, Fout=5, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=5, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyChebyshev(K=10, Fout=2, use_bias=True, use_bn=True, activation=None),
            SinusodialRepresentationDense(2,activation='sine', w0=1.0),
            hp_layer.HealpyPseudoConv(p=1, Fout=2, activation=None),
            SinusodialRepresentationDense(2,activation='sine', w0=1.0),
            tf.keras.layers.Flatten(),
            tf.keras.Dense(1)]#tf.keras.layers.Activation('m_acos')]
            #tf.keras.layers.Lambda(lambda x:tf.acos(x))] 
  tf.keras.backend.clear_session()

  with strategy.scope():
    model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=20)
    batch_size = 32
    model.build(input_shape=(None, len(indices), 6))
    model.summary(110)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError()])

  if os.path.exists('checkpoint'):
    model.load_weights('weights_theta.h5')

  else:
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[LearningRateReducerCb(),es_callback]
    )

    model.save_weights('weights_theta.h5')

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
  np.save("theta_predict.npy", predictions)
  np.save("theta_true.npy", y_test)


  predictions = model.predict(x_train)
  draw_performance(y_train,predictions, 'prediction_theta_train')
