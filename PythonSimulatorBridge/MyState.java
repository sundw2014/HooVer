package PythonSimulatorBridge;

import java.util.HashMap;
import java.util.Map;

import fr.inria.plasmalab.workflow.data.simulation.InterfaceIdentifier;
import fr.inria.plasmalab.workflow.data.simulation.InterfaceState;
import fr.inria.plasmalab.workflow.data.simulation.InterfaceTransition;
import fr.inria.plasmalab.workflow.exceptions.PlasmaSimulatorException;

public class MyState implements InterfaceState {

	double unsafe_flag;
	double time;

	public MyState(double unsafe_flag, double time) {
		this.unsafe_flag = unsafe_flag;
		this.time = time;
	}

	@Override
	public String getCategory() {
		return "MyState";
	}

	@Override
	public InterfaceIdentifier[] getHeaders() {
		InterfaceIdentifier[] ret = {PythonSimulatorBridge.TIMEID, PythonSimulatorBridge.USID};
		return ret;
	}

	@Override
	public InterfaceState cloneState() {
		return new MyState(unsafe_flag, time);
	}

	@Override
	public InterfaceTransition getIncomingTransition() {
		return null;
	}

	@Override
    public Double getValueOf(InterfaceIdentifier id) throws PlasmaSimulatorException {
		if(id.equals(PythonSimulatorBridge.USID))
						return unsafe_flag;
				else if(id.equals(PythonSimulatorBridge.TIMEID))
            return time;
        else
            throw new PlasmaSimulatorException("Unknown identifier: "+id.getName());
    }

    @Override
    public Double getValueOf(String id) throws PlasmaSimulatorException {
        if(id.equals(PythonSimulatorBridge.USID.getName()))
            return unsafe_flag;
				else if(id.equals(PythonSimulatorBridge.TIMEID.getName()))
            return time;
        else
            throw new PlasmaSimulatorException("Unknown identifier: "+id);
    }

    @Override
    public void setValueOf(InterfaceIdentifier id, double value) throws PlasmaSimulatorException {
        if(id.equals(PythonSimulatorBridge.USID))
            this.unsafe_flag = value;
				else if(id.equals(PythonSimulatorBridge.TIMEID))
            this.time = value;
        else
            throw new PlasmaSimulatorException("Unknown identifier: "+id.getName());
    }

	@Override
	public String[] toStringArray() {
		String [] ret = {Double.toString(time), Double.toString(unsafe_flag)};
		return ret;
	}

	@Override
	public Map<InterfaceIdentifier, Double> getValues() {
		Map<InterfaceIdentifier, Double> ret = new HashMap<InterfaceIdentifier, Double>();
		ret.put(PythonSimulatorBridge.TIMEID,time);
		ret.put(PythonSimulatorBridge.USID,unsafe_flag);
		return ret;
	}

}
