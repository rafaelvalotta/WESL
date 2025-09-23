# WESL
Welcome to WESL (Wind Energy Systems Lab) Optimizer. WESL is built on top of OpenMDAO to perform wind farm optimization, as well as integration with marine energy technologies such as wave and tidal energy. So far, WESL is compatible with DTU PyWake AEP calculator for wind farm flow physics computations. Additionally, WESL performs electrical cables and collection system optimization using an implementation based on Mauricio de Souza (DTU) MSc thesis supervised by Juan-Andres Perez Rua. 

Instructions to install it:

1) Download and install Anaconda in your computer to manage your Python applications. Also, choose a code editor of your preference (VSCode or similar) and install in your computer.

Anaconda: https://www.anaconda.com

VSCode: https://code.visualstudio.com


2) In Anaconda, create a new Python environment. Use Python 3.11.11 or similar.

For example: wesl_2025

3) Open a terminal and activate the new environment:

conda activate wesl_2025

4) Choose a folder to install your packages. In that folder, run:

git clone https://github.com/rafaelvalotta/WESL.git

5) After it finishs, a folder called WESL will be created. Then you run:

cd WESL

6) The last step is:

pip install -e .
