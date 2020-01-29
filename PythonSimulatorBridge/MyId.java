package PythonSimulatorBridge;

import fr.inria.plasmalab.workflow.data.simulation.InterfaceIdentifier;

public class MyId implements InterfaceIdentifier {

	String name;

	public MyId(String name) {
		this.name = name;
	}

	@Override
	public int compareTo(InterfaceIdentifier id) {
		return name.compareTo(id.getName());
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public boolean isBoolean() {
		return false;
	}

}
