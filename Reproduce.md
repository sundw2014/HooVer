We provide a docker image in which everything is setup for reproducing the results in the paper. You can access it here.
Please note that
1. Reproducing all the experiments may take a day since the implementation of the Java plugin for PlasmaLab is based on network communication and not very efficient.
2. Results of PlasmaLab may not be reproduced exactly. Since PlasmaLab doesn't provide an interface to specify the random seed for reproducibility, curves for PlasmaLab in Fig.3 may not be reproduced exactly. However, for HooVer, we use fixed random seed, and thus all the results of HooVer can be reproduced.
3. Running time of HooVer reported in Table 1 may not be reproduced, since it depends on the computational capability of the machine.

### If you want to reproduce the results in your own environment instead of docker ...
#### install requirements
In addition to the Python requirements of HooVer, we also have to install GNU parallel and JDK. For Ubuntu 18.04, the following command will setup things for you:
```
sudo apt-get install parallel openjdk-11-jdk
```
Other versions of JDK may also work, but we didn't test that. Then, we have to build the plugin for PlasmaLab.

##### Prepare PlasmaLab and the plugin
First, [download PlasmaLab](http://plasma-lab.gforge.inria.fr/download_counter.php?Download=plasma_lab_bundle/plasmalab/fr.inria.plasmalab-1.4.4-distribution.zip) and extract it. Then, download the source code of the [plugin](https://drive.google.com/file/d/1WkVKMwGa6OEw947s3ttTgZV0T0CHbCAS/view?usp=sharing) and extract it. Then, install JDK. For Ubuntu 18.04, you can run the following command to install OpenJDK-11:
```
sudo apt-get install openjdk-11-jdk
```
Other versions of JDK may also work, but we didn't test that.

Now, we can start build the plugin. First, open the build script ```PlasmaLab-plugin/PythonSimulatorBridge/build.sh``` and change the first line which should point to your plasmalab dir.
```
plasmalab_root="/home/hoover/plasmalab-1.4.4/"
```

Now, run the following command to build the plugin:
```
cd PlasmaLab-plugin/PythonSimulatorBridge/
./build.sh
```
If everything works out, the plugin should aready be created at ```[plasmalab_root]/plugins/PythonSimulatorBridge.jar```.

##### modify the scripts
Next, you may have to update the scripts for PlamaLab. Open each ```scripts/PlasmaLab_collect_data_*.py``` file, find the following line and change it to your own PlasmaLab dir:
```
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'
```

Now, you can follow the instructions on the Docker Hub page to reproduce the results.
