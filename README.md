# MIDI File generation with LSTM

This project aims to create a deep learning model that would be able to generate piano music.

## Team members
Danil Meshcherekov - d.meshcherekov@innopolis.university

## Dataset description
The main dataset used in this project is the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) that contains a collection of 200 hours of piano music recorded using a MIDI capture system.

This dataset contains everything that would be required for piano music generation: notes, times at which the note was played, and the force (loudness, or velocity) it was depressed with.

By extracting the note values, times, and velocity we get a time-series-like problem with classification (for notes) and regression (for times and velocities) that we can handle using LSTM-based neural network and several parallel fully connected layers for each sub-problem.

## Solution description
```
+--------------+      +------------------+                 1. Preprocessing step
| MIDI Dataset |----->| Extracted values |          Extracts the notes and other values
+--------------+      +------------------+         from midi files and creates a dataset
                      |   
+---------------------+----------------Neural Network--------------------------------------------------+
|                     |                                2. For each training epoch in NN                |                
|     +---------+<----+   +-------------------+      music tracks of fixed lenghts are sampled         |
|     | Samples |         | Augmented samples |       from the dataset and augmented at runtime        |
|     +---------+-------->+--------+----------+   (includes transposition, time & loudness stretching) |
|                                  |                                                                   |
|----------------------------------+-------------------------------------------------------------------+
|                                  |        3. Since the notes are categorical (discrete) values,      | 
|   +---------------------------+<-+       they are embedded first; then the embeddings are fed        |
|   | Embedding layer for notes |                           into the LSTM layer.                       |
|   |  (categorical features)   |                                                                      |
|   +-----------+---------------+    4. The output of the LSTM layer is propagated into three          |
|               |             independent parallel pipelines. The first (and most important) pipeline  |
|        +------+------+      solves the classification problem, the second one solves regression for  |
|        |  LSTM layer |                    time deltas, the third one for velocities.                 |
|        +------+------+       All three pipelines use separate optimizer: Opt. 1 backpropagates       |
|               |                              to embed, lstm layers and its FC;                       |
|               |                While opt 2 and opt 3 only affect their own FC pipelines respectively |
|               +----------------------------+------------------------------+                          |
|               |                            |                              |                          |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|     | FC + Dropout + ReLU |      | FC + Dropout + ReLU |        | FC + Dropout + ReLU |              |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|               |                            |                              |                          |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|     | FC + Dropout + ReLU |      | FC + Dropout + ReLU |        | FC + Dropout + ReLU |              |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|               |                            |                              |                          |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|     |  FC to 128 features |      |   FC to 1 feature   |        |   FC to 1 feature   |              |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|               |                            |                              |                          |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|     |    Optimizer 1      |      |    Optimizer 2      |        |    Optimizer 3      |              |
|     |  CrossEntropyLoss   |      |       L1Loss        |        |      MSELoss        |              |
|     +---------+-----------+      +---------+-----------+        +---------+-----------+              |
|               |                            |                              |                          |
+---------------+----------------------------+------------------------------+-----------------------+  |
|               +----------------------------+--+---------------------------+                          |
|                                         out   |      5. At the evaluation stage (to generate MIDI)   |
|  +----------------+     +---------------+-->--+     an empty sequence of notes is fed into NN, then  |
|  | Empty sequence |---->| Some sequence |in   |     its output is added to the sequence, and so on.  |
|  +----------------+     +---------------+--<--+       The time deltas and velocities are added to    |
|                                             separate lists, because they are not used in predictions |
+------------------------------------------------------------------------------------------------------+

```

## How to use this program
Firstly, open the root directory of this project in the terminal, run main.py.

Then, you will be prompted to create and train a new model, continue training an existing model, or use a model.

Enter the numbers from the dropping 'menu': typing '1' will ask you what model you want to train; typing '2' will ask you what model you want to use to create new MIDI file; typing '3' will exit the program.

You can see a sample use of the program below:
![Screenshot from 2023-11-28 21-10-09](https://github.com/Arloste/MusicGeneration/assets/88305350/fe2c6d69-6e98-4a28-b6bc-8cd5ba5afd69)

