package core.metrics;

import weka.core.Instance;
import core.BagSimilarity;
import core.Evaluable;
import core.InstanceDistance;
import core.VarReader;
import core.unaryOperators.Complement;


/**
 * Class for calculating the normalized Chamfer similarity between two bags. The
 * normalized Chamfer similarity is the complement of the normalized Chamfer distance.
 * The computation finally relies in an instance distance function.
 *
 * @author Sarah
 */
public class NormChamferSimilarity extends BagSimilarity {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Mathematical expression defining this function.
	   */
	  private Evaluable expression;

	  /**
	   * Creates a function which calculates the normalized Chamfer similarity between
	   * two bags which is based on the normalized Chamfer distance.
	   *
	   * @param instanceDistanceClass class of the instance distance used by the
	   * underlying normalized Chamfer distance.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public NormChamferSimilarity(Class<? extends InstanceDistance> instanceDistanceClass,
	          VarReader <Instance> X, VarReader <Instance> Y) {
	    super(X, Y);
	    expression = new Complement(new NormChamferDistance(instanceDistanceClass, X, Y));
	  }

	  /**
	   * Creates a function which calculates the normalized Chamfer similarity between
	   * two bags using the cosine distance between instances.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public NormChamferSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(CosineDistance.class, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the normalized Chamfer similarity in the
	   * current values of the arguments.
	   *
	   * @return the value of the average normalized Chamfer similarity.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }


}
