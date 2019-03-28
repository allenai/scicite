#!/bin/bash

CONDAENV=citeintent5

# Install conda
if ! (which conda) || ! [[which conda = *"miniconda"*]] || ! [[ -d $HOME/miniconda ]]; then
  echo "No conda installation found.  Installing..."
  if [[ $(uname) == "Darwin" ]]; then
    wget -nc --continue https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b || true
  else
    wget -nc --continue https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b || true
  fi
  export PATH=$HOME/miniconda/bin:$HOME/miniconda3/bin:$PATH
fi

> source $HOME/miniconda3/bin/deactivate ${CONDAENV}

conda remove -y --name ${CONDAENV} --all

conda create -n ${CONDAENV} -y python==3.6 pip pytest || true

chmod +x $HOME/miniconda3/bin/activate

echo "Activating Conda Environment ----->"
> source $HOME/miniconda3/bin/activate ${CONDAENV}

pip install -r requirements.in -c constraints.txt

python -m spacy download en
