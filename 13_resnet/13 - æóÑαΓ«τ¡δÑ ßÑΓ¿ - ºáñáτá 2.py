from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
 
 
def load_train(path):

    train_datagen = ImageDataGenerator(rescale=1/255., horizontal_flip=True, vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2)

    train_datagen_flow = train_datagen.flow_from_directory(path, target_size=(150, 150), batch_size=16, class_mode='sparse', seed=12345)

    return train_datagen_flow

 
def create_model(input_shape):

    optimizer=Adam(lr=0.005)

    model = Sequential()

#сверточный слой 
    model.add(Conv2D(6, kernel_size=(3, 3), padding='same', activation="relu", input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
#еще сверточный слой 
    model.add(Conv2D(12, kernel_size=(3, 3), padding='same', activation="relu", input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
#еще сверточный слой 
    model.add(Conv2D(18, kernel_size=(3, 3), padding='same', activation="relu", input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 
    return model 

 
def train_model(model, train_data, test_data, epochs=10, steps_per_epoch=None, validation_steps=None, batch_size=0):
    if steps_per_epoch is None:
        steps_per_epoch=len(train_data)
 
    if validation_steps is None:
        validation_steps=len(test_data)
 
    model.fit(train_data, validation_data=test_data, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=2, epochs=epochs)
 
    return model
















лог обучения: 
2022-03-29 12:07:52.625857: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
2022-03-29 12:07:53.443046: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
Using TensorFlow backend.
Found 23397 images belonging to 12 classes.
Found 7804 images belonging to 12 classes.
2022-03-29 12:08:28.278171: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2022-03-29 12:08:28.922089: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2022-03-29 12:08:28.922178: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:28.922213: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:08:29.254103: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2022-03-29 12:08:29.340165: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2022-03-29 12:08:29.981056: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2022-03-29 12:08:30.185445: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2022-03-29 12:08:30.185586: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2022-03-29 12:08:30.189914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2022-03-29 12:08:30.190317: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2022-03-29 12:08:30.454516: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2099990000 Hz
2022-03-29 12:08:30.457301: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x437f370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-03-29 12:08:30.457342: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-03-29 12:08:30.898337: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x417c8d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2022-03-29 12:08:30.898374: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0
2022-03-29 12:08:30.900878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2022-03-29 12:08:30.900940: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:30.900952: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:08:30.900983: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2022-03-29 12:08:30.900993: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2022-03-29 12:08:30.901003: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2022-03-29 12:08:30.901011: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2022-03-29 12:08:30.901019: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2022-03-29 12:08:30.905514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2022-03-29 12:08:30.911336: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:37.471932: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-03-29 12:08:37.471976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2022-03-29 12:08:37.471985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
2022-03-29 12:08:37.493641: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30509 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8b:00.0, compute capability: 7.0)
<class 'tensorflow.python.keras.engine.sequential.Sequential'>
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
Train for 1463 steps, validate for 488 steps
Epoch 1/10
2022-03-29 12:09:06.254765: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:09:10.862311: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
1463/1463 - 984s - loss: 0.9215 - acc: 0.6658 - val_loss: 0.6122 - val_acc: 0.7819
Epoch 2/10
1463/1463 - 247s - loss: 0.5354 - acc: 0.8114 - val_loss: 0.3998 - val_acc: 0.8649
Epoch 3/10
1463/1463 - 247s - loss: 0.4916 - acc: 0.8263 - val_loss: 0.3313 - val_acc: 0.8802
Epoch 4/10
1463/1463 - 246s - loss: 0.4476 - acc: 0.8449 - val_loss: 0.5692 - val_acc: 0.7668
Epoch 5/10
2022-03-29 12:07:52.625857: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
2022-03-29 12:07:53.443046: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
Using TensorFlow backend.
Found 23397 images belonging to 12 classes.
Found 7804 images belonging to 12 classes.
2022-03-29 12:08:28.278171: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2022-03-29 12:08:28.922089: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2022-03-29 12:08:28.922178: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:28.922213: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:08:29.254103: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2022-03-29 12:08:29.340165: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2022-03-29 12:08:29.981056: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2022-03-29 12:08:30.185445: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2022-03-29 12:08:30.185586: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2022-03-29 12:08:30.189914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2022-03-29 12:08:30.190317: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2022-03-29 12:08:30.454516: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2099990000 Hz
2022-03-29 12:08:30.457301: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x437f370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-03-29 12:08:30.457342: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-03-29 12:08:30.898337: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x417c8d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2022-03-29 12:08:30.898374: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0
2022-03-29 12:08:30.900878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2022-03-29 12:08:30.900940: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:30.900952: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:08:30.900983: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2022-03-29 12:08:30.900993: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2022-03-29 12:08:30.901003: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2022-03-29 12:08:30.901011: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2022-03-29 12:08:30.901019: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2022-03-29 12:08:30.905514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2022-03-29 12:08:30.911336: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2022-03-29 12:08:37.471932: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-03-29 12:08:37.471976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2022-03-29 12:08:37.471985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
2022-03-29 12:08:37.493641: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30509 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8b:00.0, compute capability: 7.0)
<class 'tensorflow.python.keras.engine.sequential.Sequential'>
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
Train for 1463 steps, validate for 488 steps
Epoch 1/10
2022-03-29 12:09:06.254765: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2022-03-29 12:09:10.862311: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
1463/1463 - 984s - loss: 0.9215 - acc: 0.6658 - val_loss: 0.6122 - val_acc: 0.7819
Epoch 2/10
1463/1463 - 247s - loss: 0.5354 - acc: 0.8114 - val_loss: 0.3998 - val_acc: 0.8649
Epoch 3/10
1463/1463 - 247s - loss: 0.4916 - acc: 0.8263 - val_loss: 0.3313 - val_acc: 0.8802
Epoch 4/10
1463/1463 - 246s - loss: 0.4476 - acc: 0.8449 - val_loss: 0.5692 - val_acc: 0.7668
Epoch 5/10
1463/1463 - 247s - loss: 0.4185 - acc: 0.8546 - val_loss: 0.4117 - val_acc: 0.8548
Epoch 6/10
1463/1463 - 246s - loss: 0.3910 - acc: 0.8674 - val_loss: 0.2717 - val_acc: 0.9058
Epoch 7/10
1463/1463 - 247s - loss: 0.3672 - acc: 0.8763 - val_loss: 0.3015 - val_acc: 0.8995
Epoch 8/10
1463/1463 - 246s - loss: 0.3551 - acc: 0.8814 - val_loss: 0.5194 - val_acc: 0.8576
Epoch 9/10
1463/1463 - 247s - loss: 0.3343 - acc: 0.8897 - val_loss: 0.2747 - val_acc: 0.9059
Epoch 10/10
1463/1463 - 248s - loss: 0.3240 - acc: 0.8949 - val_loss: 0.2614 - val_acc: 0.9099
WARNING:tensorflow:sample_weight modes were coerced from