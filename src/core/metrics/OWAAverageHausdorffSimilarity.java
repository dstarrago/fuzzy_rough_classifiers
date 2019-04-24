/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.BagSimilarity;
import core.unaryOperators.Complement;
import core.Evaluable;
import core.InstanceDistance;
import core.owaWeighing.LinealMinWeighing;
import core.MinWeighingMaker;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for calculating the Average Hausdorff similarity between two bags using 
 * OWA max and min operators. The OWA Average Hausdorff similarity is the complement
 * of the OWA Average Hausdorff distance. The computation finally relies in an instance
 * distance function. 
 *
 * @author Danel
 */
public class OWAAverageHausdorffSimilarity extends BagSimilarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Mathematical expression defining this function.
   */
  private Evaluable expression;

  /**
   * Creates a function which calculates the OWA Average Hausdorff similarity
   * between two bags.
   *
   * @param minWeighing weighing version of the min OWA operator.
   * @param instanceDistanceClass class of the instance distance used by the
   * underlying OWA average Hausdorff distance.
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public OWAAverageHausdorffSimilarity(MinWeighingMaker minWeighing,
          Class<? extends InstanceDistance> instanceDistanceClass,
          VarReader <Instance> X, VarReader <Instance> Y) {
    super(X, Y);
    expression = new Complement(new OWAAverageHausdorffDistance(minWeighing,
            instanceDistanceClass, X, Y));
  }

  /**
   * Creates a function which calculates the OWA Average Hausdorff similarity between
   * two bags using the cosine distance between instances.
   *
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public OWAAverageHausdorffSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
    this(new LinealMinWeighing(), CosineDistance.class, X, Y);
  }
  
  public OWAAverageHausdorffSimilarity(Class<? extends InstanceDistance> instanceDistanceClass,
		  VarReader <Instance> X, VarReader <Instance> Y) {
	    this(new LinealMinWeighing(), instanceDistanceClass, X, Y);
  }

  /**
   * Evaluates the expression that defines the OWA average Hausdorff similarity in the
   * current values of the arguments.
   *
   * @return the value of the OWA average Hausdorff similarity.
   */
  @Override
  public double evaluate() {
    return expression.evaluate();
  }

}
