package core.metrics;

import weka.core.Instance;
import core.BagSimilarity;
import core.Evaluable;
import core.InstanceDistance;
import core.MinWeighingMaker;
import core.VarReader;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;

/**
 * Class for calculating the normalized Chamfer similarity between two bags using 
 * OWA max and min operators. The OWA normalized Chamfer similarity is the complement
 * of the OWA normalized Chamfer distance. The computation finally relies in an instance
 * distance function. 
 *
 * @author Sarah
 */
public class OWANormChamferSimilarity extends BagSimilarity {

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
	   * @param instanceDistanceClass class of the instance distance used by the
	   * underlying OWA normalized Chamfer distance.
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public OWANormChamferSimilarity(MinWeighingMaker minWeighing,
	          Class<? extends InstanceDistance> instanceDistanceClass,
	          VarReader <Instance> X, VarReader <Instance> Y) {
	    super(X, Y);
	    expression = new Complement(new OWANormChamferDistance(minWeighing,
	            instanceDistanceClass, X, Y));
	  }

	  /**
	   * Creates a function which calculates the OWA normalized Chamfersimilarity between
	   * two bags using the cosine distance between instances.
	   *
	   * @param X variable for the first bag.
	   * @param Y variable for the second bag.
	   */
	  public OWANormChamferSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
	    this(new LinealMinWeighing(), CosineDistance.class, X, Y);
	  }
	  

	  public OWANormChamferSimilarity(Class<? extends InstanceDistance> instanceDistanceClass, 
			  VarReader <Instance> X, VarReader <Instance> Y) {
	    this(new LinealMinWeighing(), instanceDistanceClass, X, Y);
	  }

	  /**
	   * Evaluates the expression that defines the OWA normalized Chamfer similarity in the
	   * current values of the arguments.
	   *
	   * @return the value of the OWA normalized Chamfer similarity.
	   */
	  @Override
	  public double evaluate() {
	    return expression.evaluate();
	  }

}
