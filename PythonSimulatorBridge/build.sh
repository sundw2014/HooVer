plasmalab_root="/home/daweis2/plasmalab-1.4.4/"
third_party_root="/home/daweis2/SMC_MDP/PythonSimulationBridge_build/"
javac -Xlint:deprecation -cp ${plasmalab_root}/libs/fr.inria.plasmalab.algorithm-1.4.4.jar:${plasmalab_root}/libs/fr.inria.plasmalab.workflow-1.4.4.jar:${plasmalab_root}/libs/jspf.core-1.0.2.jar:${plasmalab_root}/libs/lf4j-api-1.7.16.jar:${third_party_root}/jeromq-0.5.1.jar:${third_party_root}/msgpack-0.6.2.jar *.java
cd ${third_party_root}
jar cvf PythonSimulatorBridge.jar . 1>/dev/null
mv PythonSimulatorBridge.jar ${plasmalab_root}/plugins/
cd -
