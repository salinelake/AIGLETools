<p align="center" >
  <img width="80%" src="toypolymer.png" />
</p>

# Using AIGLE and AILE to model the backbone dynamics of a toy polymer.

### Step 1: generate_data.ipynb
Using OPENMM to generate a 10ns-long trajectory data of the toy polymer

### Step 2: train_AIGLE.ipynb
Train the AIGLE model. GPU is recommended. Otherwise calculating correlation function takes a long time.
The model configuration has been stored in the "gle_paras" folder.

### Step 3: simulate.ipynb
Simulate AIGLE and AILE with OPENMM for 100ns

### Step 4: compare_MD_GLE_LE.ipynb
Compare MD, AIGLE and AILE.