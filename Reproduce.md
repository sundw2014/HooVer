We provide a VirtualBox VM image in which everything is setup for reproducing the results in the paper. You can download the VM [here](). We tested this docker image on a Linux workstation with two Xeon Silver 4110 CPUs and 32 GB RAM. You may have to change the configuration in VirtualBox to fit your host machine.

Please note that
1. Reproducing all the experiments may take a day since the implementation of the Java plugin for PlasmaLab is based on network communication and not very efficient.
2. Results of PlasmaLab may not be reproduced exactly. Since PlasmaLab doesn't provide an interface to specify the random seed for reproducibility, curves for PlasmaLab in Fig.3 may not be reproduced exactly. However, for HooVer, we use fixed random seeds, and thus all the results of HooVer can be reproduced exactly.
3. Running time of HooVer reported in Table 1 may not be reproduced, since it depends on the computational capability of the host machine.

### Reproduce the results in VM
You can just run the following command in the VM to reproduce all the results:
```
cd HooVer/scripts
./reproduce.sh
```
This may take 2 days to finish due to slow communication between PlasmaLab and the external simulator. After finishing, the generated data will be located in ```HooVer/data``` and the following files will be generated in ```HooVer/scripts```:
```
Slplatoon3.pdf
Mlplatoon.pdf
DetectingPedestrian.pdf
Merging.pdf
Conceptual
Table1.txt
Table2.txt
```

### If you want to reproduce the results in your own environment instead of docker ...
#### install requirements
In addition to the Python requirements of HooVer, we also have to install GNU parallel and JDK. For Ubuntu 18.04, the following command will setup things for you:
```
sudo apt-get install parallel openjdk-11-jdk
```
Other versions of JDK may also work, but we didn't test that. Then, we have to build the plugin for PlasmaLab.

##### Prepare PlasmaLab and the plugin
First, [download PlasmaLab](http://plasma-lab.gforge.inria.fr/download_counter.php?Download=plasma_lab_bundle/plasmalab/fr.inria.plasmalab-1.4.4-distribution.zip) and extract it. Then, [download the source code](https://drive.google.com/file/d/1WkVKMwGa6OEw947s3ttTgZV0T0CHbCAS/view?usp=sharing) of the plugin and extract it.

Now, we can start build the plugin. First, open the build script ```PlasmaLab-plugin/PythonSimulatorBridge/build.sh``` and find the following line and change it to your own plasmalab dir:
```
plasmalab_root="/home/daweis2/plasmalab-1.4.4/"
```

Now, run the following command to build the plugin:
```
cd PlasmaLab-plugin/PythonSimulatorBridge/
./build.sh
```
If everything works out, the plugin should already be created at ```[plasmalab_root]/plugins/PythonSimulatorBridge.jar```.

##### update the scripts
Next, you may have to update the scripts for PlasmaLab. Open each ```scripts/PlasmaLab_collect_data_*.py``` file, find the following line and change it to your own PlasmaLab dir:
```
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'
```
