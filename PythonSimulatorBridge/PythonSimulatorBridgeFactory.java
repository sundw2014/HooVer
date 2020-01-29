package PythonSimulatorBridge;

import java.io.File;

import fr.inria.plasmalab.workflow.data.AbstractModel;
import fr.inria.plasmalab.workflow.data.factory.AbstractModelFactory;
import fr.inria.plasmalab.workflow.exceptions.PlasmaDataException;
import net.xeoh.plugins.base.annotations.PluginImplementation;

@PluginImplementation
public class PythonSimulatorBridgeFactory extends AbstractModelFactory {

	private final static String id = "PythonSimulatorBridge";

	@Override
	public String getName() {
		return "PythonSimulatorBridge";
	}

	@Override
	public String getDescription() {
		return "This is a tutorial simulator plugin";
	}

	@Override
	public String getId() {
		return id;
	}


	@Override
	public AbstractModel createAbstractModel(String name) {
		return new PythonSimulatorBridge(name,"", id);
	}

	@Override
	public AbstractModel createAbstractModel(String name, File file) throws PlasmaDataException {
		return new PythonSimulatorBridge(name,file,id);
	}

	@Override
	public AbstractModel createAbstractModel(String name, String content) {
		return new PythonSimulatorBridge(name,content,id);
	}


}
