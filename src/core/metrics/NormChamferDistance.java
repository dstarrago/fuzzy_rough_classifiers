package core.metrics;

import java.lang.reflect.Constructor;

import weka.core.Instance;
import core.BagDistance;
import core.Const;
import core.Evaluable;
import core.InstanceDistance;
import core.SizeOfBag;
import core.Var;
import core.VarReader;
import core.binaryOperators.Div2;
import core.binaryOperators.Sum2;
import core.iterators.InstancesFromBag;
import core.multipleOperators.Min;
import core.multipleOperators.Sum;


/**
 * Class for calculating the normalized Chamfer distance between two bags. 
 * This is the usual Chamfer distance divided by 2.
 * The computation finally relies in an instance distance function.
 *
 * @author Sarah
 */
public class NormChamferDistance extends BagDistance {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Mathematical expression defining this function.
	   */
	  private Evaluable expression;

	  /**
	   * Creates a function which calculates the normalized Chamfer distance between two
	   * bags. This is the Chamfer distance divided by 2.
	   *
	   * @param instanceDistanceClass class of the instance distance used.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public NormChamferDistance(Class<? extends InstanceDistance> instanceDistanceClass,
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
			    				  new Div2(new Sum(new InstancesFromBag(x, X), new Min(new InstancesFromBag(y, Y), instanceDistance)), new SizeOfBag(X)),
			    				  new Div2(new Sum(new InstancesFromBag(y, Y), new Min(new InstancesFromBag(x, X), instanceDistance)), new SizeOfBag(Y))
			    		  ),
			    		  new Const(2.0)
			    	   );
	    } catch (Exception e) { 
	      System.out.println("Problema al construir ChamferDistance: " + e.toString());
	    }
	  }

	  /**
	   * Creates a function which calculates the normalized distance between
	   * two bags using the cosine distance between instances.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public NormChamferDistance(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(CosineDistance.class, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the normalized Chamfer distance in the
	   * current values of the arguments.
	   *
	   * @return the value of the normalized Chamfer distance.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }

}
