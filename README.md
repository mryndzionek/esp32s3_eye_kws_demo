# esp32s3_eye_kws_demo

Speech recognition is based on [this](https://github.com/microsoft/EdgeML/blob/master/docs/publications/Sha-RNN.pdf)
model and examples from the same repository. The cell type in this model is [FastRNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/FastGRNN.pdf).
The inference with this model takes around 140ms. The inference is run every 250ms, so four times a second.

A bigger, LSTM-based model with ~550ms inference time can be found [here](https://github.com/mryndzionek/esp32s3_eye_kws_demo/tree/lstm_model).
It is slightly more accurate, especially to the `up` label.

https://github.com/user-attachments/assets/861b4d5a-1f38-4653-9b4f-e0f713c1e0ba
