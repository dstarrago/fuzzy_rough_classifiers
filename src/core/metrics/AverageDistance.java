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
import core.binaryOperators.Mult2;
import core.binaryOperators.Sum2;
import core.iterators.InstancesFromBag;
import core.multipleOperators.Min;
import core.multipleOperators.Sum;

/**
 * Class for calculating the average distance between two bags. The
 * computation finally relies in an instance distance function.
 *
 * @author Sarah
 */
public class AverageDistance extends BagDistance {
	
	 /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Mathematical expression defining this function.
	   */
	  private Evaluable expression;

	  /**
	   * Creates a function which calculates the Average distance between
	   * two bags.
	   *
	   * @param instanceDistanceClass class of the instance distance used.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public AverageDistance(Class<? extends InstanceDistance> instanceDistanceClass,
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
			              new Sum(new InstancesFromBag(x, X), new Sum(new InstancesFromBag(y, Y), instanceDistance)),	              
			              new Mult2(new SizeOfBag(X), new SizeOfBag(Y))
				  	   );

	    } catch (Exception e) { }
	  }

	  /**
	   * Creates a function which calculates the Average distance between
	   * two bags using the cosine distance between instances.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public AverageDistance(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(CosineDistance.class, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the average distance in the
	   * current values of the arguments.
	   *
	   * @return the value of the average distance.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }

}
