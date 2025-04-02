Selective Inference for DNN example

Example: Simple Anomaly Detection after DNN-based Feature Extracting

Anomaly Detection: The one with the largest sum is choosen as anomaly

Feature Extractor: Just treat it as a neuron network converting X (d-dimension) -> X' (d'-dimension), don't need to completely understand it

Test statistic from: https://arxiv.org/abs/2310.14608

Focus on the function get_dnn_interval in util.py
