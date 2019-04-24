/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.BagSimilarity;
import core.unaryOperators.Complement;
import core.Evaluable;
import core.InstanceDistance;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.MaxWeighingMaker;
import core.MinWeighingMaker;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for calculating the Hausdorff similarity between two bags using OWA max 
 * and min operators. The OWA Hausdorff similarity is the complement of the OWA
 * Hausdorff distance. The computation finally relies in an instance distance function.
 *
 * @author Danel
 */
public class OWAHausdorffSimilarity extends BagSimilarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Mathematical expression defining this function.
   */
  private Evaluable expression;

  /**
   * Creates a function which calculates the OWA Hausdorff similarity between two
   * bags.
   *
   * @param maxWeighing weighing version of the max OWA operator.
   * @param minWeighing weighing version of the min OWA operator.
   * @param instanceDistanceClass class of the instance distance used by the
   * underlying OWA Hausdorff distance.
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public OWAHausdorffSimilarity(MaxWeighingMaker maxWeighing, MinWeighingMaker minWeighing,
          Class<? extends InstanceDistance> instanceDistanceClass,
          VarReader <Instance> X, VarReader <Instance> Y) {
    super(X, Y);
    expression = new Complement( new OWAHausdorffDistance(maxWeighing, minWeighing,
            instanceDistanceClass, X, Y));
  }

  /**
   * Creates a function which calculates the OWA Hausdorff similarity between
   * two bags using the cosine distance between instances and the lineal weighing
   * versions of the max and min OWA operators.
   *
   * @param X variable for the first instance.
   * @param Y variable for the second instance.
   */
  public OWAHausdorffSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
    this(new LinealMaxWeighing(), new LinealMinWeighing(), CosineDistance.class, X, Y);
  }
  
  public OWAHausdorffSimilarity(Class<? extends InstanceDistance> instanceDistanceClass,
		  VarReader <Instance> X, VarReader <Instance> Y) {
	    this(new LinealMaxWeighing(), new LinealMinWeighing(), instanceDistanceClass, X, Y);
	  }

  /**
   * Evaluates the expression that defines the OWA Hausdorff similarity in the
   * current values of the arguments.
   *
   * @return the value of the OWA Hausdorff similarity.
   */
  @Override
  public double evaluate() {
    return expression.evaluate();
  }

}
