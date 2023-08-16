conda install python=3.9.1
conda install virtualenv
virtualenv venv
python -m venv venv

source ./venv/bin/activate

pip install --upgrade pip
pip install numpy
pip install scipy
pip install lib5c
pip install matplotlib
