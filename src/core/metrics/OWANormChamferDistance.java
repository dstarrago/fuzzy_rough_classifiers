package core.metrics;

import java.lang.reflect.Constructor;

import weka.core.Instance;
import core.BagDistance;
import core.Const;
import core.Evaluable;
import core.InstanceDistance;
import core.MinWeighingMaker;
import core.SizeOfBag;
import core.Var;
import core.VarReader;
import core.binaryOperators.Div2;
import core.binaryOperators.Sum2;
import core.iterators.InstancesFromBag;
import core.multipleOperators.Min;
import core.multipleOperators.OWA;
import core.multipleOperators.Sum;
import core.owaWeighing.LinealMinWeighing;


/**
 * Class for calculating the normalized Chamfer similarity between two bags using 
 * OWA max and min operators. The computation finally relies in an instance
 * similarity function. 
 *
 * @author Sarah
 */
public class OWANormChamferDistance extends BagDistance {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Mathematical expression defining this function.
	   */
	  private Evaluable expression;

	  /**
	   * Creates a function which calculates the OWA normalized Chamfer similarity
	   * between two bags.
	   *
	   * @param minWeighing weighing version of the min OWA operator.
	   * @param instanceDistanceClass class of the instance distance used.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public OWANormChamferDistance(MinWeighingMaker minWeighing,
	          Class<? extends InstanceDistance> instanceDistanceClass,
	          VarReader <Instance> X, VarReader <Instance> Y) {
	    super(X, Y);
	    Var<Instance> x = new Var();
	    Var<Instance> y = new Var();
	    try {
	      /**
	       * Instance Distance Constructor
	       */
	      Constructor<? extends InstanceDistance> cons =
	              instanceDistanceClass.getConstructor(VarReader.class, VarReader.class);
	      InstanceDistance instanceDistance = cons.newInstance(x, y);
	      
	      expression = new Div2( 
	    		  new Sum2(
	    				  new Div2(new Sum(new InstancesFromBag(x, X), new OWA(minWeighing, new InstancesFromBag(y, Y), instanceDistance)), new SizeOfBag(X)),
	    				  new Div2(new Sum(new InstancesFromBag(y, Y), new OWA(minWeighing, new InstancesFromBag(x, X), instanceDistance)), new SizeOfBag(Y))
	    		  ),
	    		  new Const(2.0)
	    		  );
          } catch (Exception e) { }
	  }

	  /**
	   * Creates a function which calculates the OWA normalized Chamfer distance between
	   * two bags using the cosine distance between instances and the linear weighing
	   * versions of the min OWA operator.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public OWANormChamferDistance(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(new LinealMinWeighing(), CosineDistance.class, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the OWA normalized Chamfer distance in the
	   * current values of the arguments.
	   *
	   * @return the value of the OWA normalized Chamfer distance.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }

}
