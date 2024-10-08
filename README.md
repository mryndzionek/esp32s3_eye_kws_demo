# esp32s3_eye_kws_demo

Speech recognition is based on [this](https://github.com/microsoft/EdgeML/blob/master/docs/publications/Sha-RNN.pdf)
architecture and examples from the same repository. The cell type in this model is [FastGRNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/FastGRNN.pdf).
More detailed view on data flow through the network with specific vector/matrix sizes:

![sharnn](images/sharnn.png)

The inference is run nine times a second. The CPU utilization due to inference is only ~24%.
FastRNN cell is also supported (can be changed via `menuconfig`).

A bigger, LSTM-based model with ~550ms inference time can be found [here](https://github.com/mryndzionek/esp32s3_eye_kws_demo/tree/lstm_model).
It is slightly more accurate, especially to the `up` label.

https://github.com/user-attachments/assets/861b4d5a-1f38-4653-9b4f-e0f713c1e0ba


## Notes

Number of TinyML model conversion frameworks were tested,
but none gave satisfactory results. The main problem seems
to be that the graphs exported from PyTorch (or other
training-oriented NN frameworks) contain much additional
information needed only for training, but information
which obscures the essential structure needed only for inference.
Here is for example a ONNX graph exported directly from PyTorch:

![graph](images/pytorch_graph.png)

and [this](https://github.com/mryndzionek/esp32s3_eye_kws_demo/blob/main/main/fast_grnn.c) is
all the "manually-transpiled" code needed for inference (~170 LoCs of C) ...

