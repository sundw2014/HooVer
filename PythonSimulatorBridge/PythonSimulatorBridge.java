package PythonSimulatorBridge;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
// import java.io.*;
// import org.apache.commons.io.FilenameUtils;

// import org.python.util.PythonInterpreter;
// import org.python.core.PyObject;
// import org.python.core.PyList;
// import org.python.core.PyFloat;
// import org.python.core.PyLong;
// import org.python.core.PyObject;

import fr.inria.plasmalab.workflow.data.AbstractModel;
import fr.inria.plasmalab.workflow.data.simulation.InterfaceIdentifier;
import fr.inria.plasmalab.workflow.data.simulation.InterfaceState;
import fr.inria.plasmalab.workflow.exceptions.PlasmaDataException;
import fr.inria.plasmalab.workflow.exceptions.PlasmaDeadlockException;
import fr.inria.plasmalab.workflow.exceptions.PlasmaSimulatorException;
import fr.inria.plasmalab.workflow.exceptions.PlasmaSyntaxException;

import org.zeromq.SocketType;
import org.zeromq.ZMQ;
import org.zeromq.ZContext;
import org.msgpack.MessagePack;
// import org.msgpack.annotation.Message;
import org.msgpack.template.Templates;

// FIXME: various number of state variables
public class PythonSimulatorBridge extends AbstractModel {

	protected static final MyId USID = new MyId("US"); //VALUE
    protected static final MyId TIMEID = new MyId("T"); //TIME

	BufferedReader br;
	MyState initialState;
	List<InterfaceState> trace;
	ZContext context = new ZContext();
	ZMQ.Socket socket;
	MessagePack msgpack = new MessagePack();

	double [] internal_state;
	int T_MAX;
        int port;

	// PythonInterpreter interpreter = new PythonInterpreter();
	// PyObject pysimulate;

	public PythonSimulatorBridge(String name, String content, String id) {
		this.name = name;
		this.content = content;
		this.errors = new ArrayList<PlasmaDataException>();
		this.origin = null;
		this.id = id;
	}

	public PythonSimulatorBridge(String name, File file, String id) throws PlasmaDataException {
		this.name = name;
		this.content = "";
		this.errors = new ArrayList<PlasmaDataException>();
		this.origin = file;
		this.id = id;
		try {
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);
			while(br.ready())
				content = content+br.readLine();
			br.close();
		} catch (IOException e) {
			throw new PlasmaDataException("Cannot read model file",e);
		}
	}

	@Override
	public boolean checkForErrors() {
		// Empty from previous errors
		errors.clear();
		try{
			final Integer[] ints = Arrays.stream(content.split(" "))
				.map(Integer::parseInt)
				.toArray(Integer[]::new);
			if(ints.length != 3){
				errors.add(new PlasmaDataException("need three ints, num_states, T_MAX, port"));
			}
			int num_states = ints[0];
			T_MAX = ints[1];
                        port = ints[2];
			internal_state = new double [num_states+3];
		} catch(Exception e) {
			errors.add(new PlasmaDataException("Cannot read model content",e));
		}
		socket = context.createSocket(SocketType.REQ);
	 	//  Socket to talk to server
	 	System.out.println("Connecting to server...");
	 	socket.connect("tcp://localhost:"+Integer.toString(port));

		return !errors.isEmpty();
	}

	@Override
	public void setValueOf(Map<InterfaceIdentifier, Double> update) throws PlasmaSimulatorException {
		// This method change the initial state with a set of initial values.
	    for(InterfaceIdentifier id:update.keySet())
	    	initialState.setValueOf(id, update.get(id));
	}

	@Override
	public InterfaceState newPath() {
		return newPath(0);
	}

	@Override
	public InterfaceState newPath(List<InterfaceState> arg0) {
		return newPath(0);
	}

	@Override
	public InterfaceState newPath(long arg0) {
		trace = new ArrayList<InterfaceState>();

		// Create serialize objects.
		// set time = -1, unsafe_flag, and seed
		internal_state[internal_state.length-3] = -1;
		internal_state[internal_state.length-2] = 0;
                long seed = arg0 % 4294967296l;
		internal_state[internal_state.length-1] = (double) (seed);

		System.out.println("newPath: seed = " + String.valueOf(seed));

		List<Double> state = new ArrayList<Double>();
		for (int i = 0; i < internal_state.length; i++)
			state.add(internal_state[i]);

		// Serialize
		try{
			byte[] bytes = msgpack.write(state);
			// System.out.println("Sending");
			socket.send(bytes, 0);
		}catch(IOException ex){}

		byte[] reply = socket.recv(0);
		// Deserialize directly using a template
		try{
			state = msgpack.read(reply, Templates.tList(Templates.TDouble));
		}catch(IOException ex){}

		// System.out.println("Received " + state.get(0) + ", " + state.get(1) + ", " + state.get(2) + ", " + state.get(3) + ", " + state.get(4));
		trace.add(new MyState(state.get(state.size()-1), state.get(state.size()-2)));
		for (int i = 0; i < internal_state.length-1; i++)
			internal_state[i] = state.get(i);
    return getCurrentState();
	}

	@Override
    public InterfaceState simulate() throws PlasmaSimulatorException {
			MyState currentState = (MyState) getCurrentState();
			if (currentState.time >= T_MAX)
				throw new PlasmaDeadlockException(getCurrentState(), getTraceLength());
			// Create serialize objects.
			// set time = 0, unsafe_flag, and seed
			List<Double> state = new ArrayList<Double>();
			for (int i = 0; i < internal_state.length; i++)
				state.add(internal_state[i]);

			// Serialize
			try{
				byte[] bytes = msgpack.write(state);
				// System.out.println("Sending");
				socket.send(bytes, 0);
			}catch(IOException ex){}

			byte[] reply = socket.recv(0);
			// Deserialize directly using a template
			try{
				state = msgpack.read(reply, Templates.tList(Templates.TDouble));
			}catch(IOException ex){}

			// System.out.println("Received " + state.get(0) + ", " + state.get(1) + ", " + state.get(2) + ", " + state.get(3) + ", " + state.get(4));
			trace.add(new MyState(state.get(state.size()-1), state.get(state.size()-2)));
			for (int i = 0; i < internal_state.length-1; i++)
				internal_state[i] = state.get(i);

	    return getCurrentState();
    }

	@Override
	public void backtrack() throws PlasmaSimulatorException {
		throw new PlasmaSimulatorException("Not implemented");
	}

	@Override
	public void cut(int arg0) throws PlasmaSimulatorException {
		throw new PlasmaSimulatorException("Not implemented");
	}

	@Override
	public InterfaceState getCurrentState() {
		return trace.get(getTraceLength()-1);
	}

	@Override
	public int getDeadlockPos() {
		return getTraceLength()-1;
	}

	@Override
	public InterfaceIdentifier[] getHeaders() {
		InterfaceIdentifier[] ret = {PythonSimulatorBridge.TIMEID, PythonSimulatorBridge.USID};
		return ret;
	}

	@Override
	public Map<String, InterfaceIdentifier> getIdentifiers() {
		Map<String, InterfaceIdentifier> mymap = new HashMap<String, InterfaceIdentifier>();
		mymap.put(TIMEID.getName(), TIMEID);
		mymap.put(USID.getName(), USID);
		return mymap;
	}

	@Override
	public InterfaceState getStateAtPos(int index) {
		return trace.get(index);
	}

	@Override
	public List<InterfaceIdentifier> getStateProperties() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public InterfaceIdentifier getTimeId() {
		return TIMEID;
	}

	@Override
	public List<InterfaceState> getTrace() {
		return trace;
	}

	@Override
	public boolean hasTime() {
		return true;
	}
}
