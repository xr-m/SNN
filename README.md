# SNN
the code for Spiking neural networks (SSN) on EEG emotion recognition

**reference paper**: Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network

**datasets**: SEED (3 classes), DEAP-Arousal (2 classes), DEAP-Valence (2 classes).

**data preprocessing**: please run tools/process_deap.py or tools/process_seed.py

**the file train.py involves 3 modes**:

cfg_mode = 'deapA'; cfg_mode = 'deapV'; cfg_mode = 'seed';

you can choose one mode to run train.py directly.
