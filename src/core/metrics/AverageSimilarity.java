package core.metrics;

import weka.core.Instance;
import core.BagSimilarity;
import core.Evaluable;
import core.InstanceDistance;
import core.VarReader;
import core.unaryOperators.Complement;


/**
 * Class for calculating the Average similarity between two bags. The
 * Average similarity is the complement of the Average distance.
 * The computation finally relies in an instance distance function.
 *
 * @author Sarah
 */
public class AverageSimilarity extends BagSimilarity {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Mathematical expression defining this function.
	   */
	  private Evaluable expression;

	  /**
	   * Creates a function which calculates the Average similarity between
	   * two bags which is based on the average distance.
	   *
	   * @param instanceDistanceClass class of the instance distance used by the
	   * underlying average distance.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public AverageSimilarity(Class<? extends InstanceDistance> instanceDistanceClass,
	          VarReader <Instance> X, VarReader <Instance> Y) {
	    super(X, Y);
	    expression = new Complement(new AverageDistance(instanceDistanceClass, X, Y));
	  }

	  /**
	   * Creates a function which calculates the Average similarity between
	   * two bags using the cosine distance between instances.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public AverageSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(CosineDistance.class, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the average similarity in the
	   * current values of the arguments.
	   *
	   * @return the value of the average similarity.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }

}
