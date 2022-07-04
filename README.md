# tetris-rl

A Python implementation of Tetris that can be run in a Jupyter notebook (with or without UI) and eventually serves as a reinforcement learning environment.

Policy gradient and deep Q learning algorithms (implemented with TensorFlow and Keras) are used to train an agent to play the game. The neural network configuration can be saved as a .h5 file.

There is also an option to play the game manually beforehand to get a better feeling of what actions there are and how the agent's reward is determined. The Tetris UI is outputted as a string. 

Parts of this project use code snippets from Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
