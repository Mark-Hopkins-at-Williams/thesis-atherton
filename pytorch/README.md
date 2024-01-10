## To create a virtual environment on appa that works with this code:

conda create --name=ENV_NAME python=3.10.4
conda activate ENV_NAME
conda install pytorch torchvision torchaudio torchtext torchdata pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -U pydantic spacy
pip install 'portalocker>=2.0.0'