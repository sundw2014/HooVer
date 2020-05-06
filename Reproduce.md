We provide a VirtualBox VM image in which everything is setup for reproducing the results in the paper. You can download the VM [here](https://drive.google.com/file/d/1AKUAgTAj9IPHdvPCeGDyEwmopQpSIfE7/view?usp=sharing)(username & passwd: hoover). We tested this image on a Linux workstation with two Xeon Silver 4110 CPUs and 32 GB RAM. You may have to change the configuration in VirtualBox to fit your host machine.

Please note that
1. Reproducing all the experiments may take about two days since the implementation of the Java plugin for PlasmaLab is based on network communication and not very efficient.
2. Results of PlasmaLab may not be reproduced exactly. Since PlasmaLab doesn't provide an interface to specify the random seed for reproducibility, curves for PlasmaLab in Fig.6 may not be reproduced exactly. However, for HooVer, we use fixed random seeds, and thus all the results of HooVer can be reproduced exactly.
3. Running time of HooVer reported in Table 1 may not be reproduced, since it depends on the computational capability of the host machine.
4. We have tested the VM and generated data and results in ```HooVer/data_old``` and ```HooVer/results_old```.

### Reproduce the results in VM
You can run the following command in the VM to reproduce all the results:
```
cd HooVer/scripts
./reproduce.sh
```
This may take 2 days to finish due to slow communication between PlasmaLab and the external simulator. After finishing, the generated data will be located in ```HooVer/data```.  The following files will be generated in ```HooVer/results```:
```
Slplatoon3.pdf
Mlplatoon18.pdf
DetectingPedestrian.pdf
Merging.pdf
Table1.txt
Table2.txt
ConceptualModel_ss_result_nqueries.pdf
```
If you want to see some results quickly, you can run the follow the following instructions to reproduce each result separately.
#### Figure 5
We evaluated HooVer on 4 benchmarks and the result is reported in Figure 5. For each benchmark:

##### SLplatoon
```
cd HooVer/scripts
./reproduce_figure1_1.sh
```
The output figure will be located at ```HooVer/results/SLplatoon.pdf```.

##### MLplatoon
```
cd HooVer/scripts
./reproduce_figure1_2.sh
```
The output figure will be located at ```HooVer/results/MLplatoon18.pdf```.

##### DetectBrake
```
cd HooVer/scripts
./reproduce_figure1_3.sh
```
The output figure will be located at ```HooVer/results/DetectingPedestrian.pdf```.

##### Merging
```
cd HooVer/scripts
./reproduce_figure1_4.sh
```
The output figure will be located at ```HooVer/results/Merging.pdf```.

#### Table 1
```
cd HooVer/scripts
./reproduce_table1.sh
```
The output figure will be located at ```HooVer/results/Table1.txt```.

#### Table 2
```
cd HooVer/scripts
./reproduce_table2.sh
```
The output figure will be located at ```HooVer/results/Table2.txt```.

#### Figure 6
```
cd HooVer/scripts
./reproduce_figure7.sh
```
The output figure will be located at ```HooVer/results/ConceptualModel_ss_result_nqueries.pdf```.


### If you want to reproduce the results in your own environment instead of using VM ...
#### Install requirements
In addition to the Python requirements of HooVer, you also have to install GNU parallel and JDK. For Ubuntu 18.04, the following command will setup things for you:
```
sudo apt-get install parallel openjdk-11-jdk
```
Other versions of JDK may also work, but we didn't test that. Then, you have to build the plugin for PlasmaLab.

##### Prepare PlasmaLab and the plugin
First, [download PlasmaLab](http://plasma-lab.gforge.inria.fr/download_counter.php?Download=plasma_lab_bundle/plasmalab/fr.inria.plasmalab-1.4.4-distribution.zip) and extract it. Then, [download the source code](https://drive.google.com/file/d/15g8Q085TfsKJNKhtAjZA_KMGT_qehAIw/view?usp=sharing) of the plugin and extract it.

Now, you can start to build the plugin. First, open the build script ```PlasmaLab-plugin/PythonSimulatorBridge/build.sh```, find the following line and change it to your own plasmalab dir:
```
plasmalab_root="/home/daweis2/plasmalab-1.4.4/"
```

Now, run the following command to build the plugin:
```
cd PlasmaLab-plugin/PythonSimulatorBridge/
./build.sh
```
If everything works out, the plugin should already be created at ```[plasmalab_root]/plugins/PythonSimulatorBridge.jar```.

##### Update the scripts
Next, you may have to update the scripts for running PlasmaLab. Open each ```scripts/PlasmaLab_collect_data_*.py``` file, find the following line and change it to your own PlasmaLab dir:
```
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'
```

Now, you can run ```scripts/reproduce.sh``` to reproduce the results.
