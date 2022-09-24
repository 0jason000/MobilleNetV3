# MobilleNetV3

Besides using inverted residual structure where the input and output of the residual block are thin bottleneck layers and lightweight depthwise convolutions of MobileNetV2, MobileNetV3 combines NAS and NetAdapt algorithm that can be transplanted  to mobile devices like mobile phones, in order to deliver the next generation of high accuracy efficient neural network models to power on-device computer vision.

The architectural definition of each network refers to the following papers:

[1] Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

## Examples

***

### Train

- The following configuration uses 8 GPUs for training and the image input size is set to 224.

  ```shell
  mpirun -n 8 python train.py --model mobilenet_v3_small --data_url ./dataset/imagenet --epoch_size 120
  ```

  output:

  ```text
  Epoch:[ 90/ 120], step:[    1/ 2502], loss:[2.771/2.771], time:177.988 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    2/ 2502], loss:[2.183/2.183], time:201.013 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    3/ 2502], loss:[2.595/2.595], time:45.759 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    4/ 2502], loss:[2.655/2.713], time:131.621 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    5/ 2502], loss:[2.767/2.475], time:133.700 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    6/ 2502], loss:[2.461/2.528], time:49.254 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    7/ 2502], loss:[2.469/2.508], time:55.803 ms, lr:0.07329
  ...
  ```

### Eval

- The following configuration for Mobilenet_v3_small eval.

  ```shell
  python validate.py --model mobilenet_v3_small --data_url ./dataset/imagenet
  ```

  output:

  ```text
  {'Top_1_Accuracy': 0.6381241997439181, 'Top_5_Accuracy': 0.8480513764404609}
  ```
  