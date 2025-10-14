# CoCCoN Workshop SBI Tutorial 2025


## Local Setup
```bash
# git clone 
# cd

# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

pip install sbi pandas
pip install --upgrade jupyter notebook ipykernel

python3 -m ipykernel install --user --name coccon_sbi \
       --display-name "CoCCoN Workshop 2025: SBI Tutorial"

jupyter notebook          # or:  jupyter lab
```