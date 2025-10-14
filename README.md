# CoCCoN Workshop SBI Tutorial 2025
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vagechirkov/coccon_workshop_2025_sbi_tutorial/blob/main/coccon_workshop_2025_sbi_tutorial.ipynb)

## Local Setup
```bash
# git clone https://github.com/vagechirkov/coccon_workshop_2025_sbi_tutorial.git
# cd coccon_workshop_2025_sbi_tutorial

# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

pip install sbi pandas
pip install --upgrade jupyter notebook ipykernel

python3 -m ipykernel install --user --name coccon_sbi \
       --display-name "CoCCoN Workshop 2025: SBI Tutorial"

jupyter lab   # or:  jupyter notebook
```