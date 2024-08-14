## DTI-MvSCA

code for " DTI-MvSCA: A multi-view neural network framework for drug-target interaction prediction"

### Quick start

**We have processed the data for data_v3 and data_v5 according to the steps described in the paper. To proceed, please unzip "DTI-MvSCA_data_processed.rar" and run "main.py".**

row data "hetero_dataset.zip“ link：https://pan.quark.cn/s/e65dd7f011e6

If you wish to process your own data, you can unzip "hetero_dataset.zip"  and open the "data_process" folder then run the files in the `./matlab`, `./DAE`, `./negative_sampler`,`./gen_dti` folders sequentially.

##### environment 

torch == 2.0.0+cu118

torch-geometric == 2.3.1 

