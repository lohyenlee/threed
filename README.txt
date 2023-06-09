#==============================================================================
# THE FOLLOWING WORKS:
#==============================================================================
conda deactivate
conda remove -n p3j --all
conda create -n p3j
conda activate p3j
conda install -c conda-forge jupyterlab=3.6.3
conda install -c conda-forge jupyterlab_widgets=3.0.5 #pythreejs fails with 3.0.6 onwards 
conda install -c conda-forge pythreejs

#==============================================================================
# SUGGESTED SETTINGS IN JUPYTER LAB
#==============================================================================
Set notebook indent=2


