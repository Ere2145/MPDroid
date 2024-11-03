1. Get FCG: Install androguard and obtain the fcg from process_dataset.py
2. Get Raw Api_Sequence: Install andropytool, it is recommended to use docker, and select the droidbox option when running to obtain dynamic features
3. Get SAG:Extract api sequence from dynamic features through sysapi_seq.py, sysAPI_to_VECc. py, sysapi_onhot.py vectorization, api_vec_to_graph to get sag(called apig in code)
4. Main: multimodal_2gcn_late_Fusion.py is the main implementation code, including fcg preprocessing, data set processing, model and training.
5. Retrain: model_2gcn_retrain is a downstream task. Input sag to obtain classification results and reverse retrain the model