/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.BagSimilarity;
import core.unaryOperators.Complement;
import core.Evaluable;
import core.InstanceDistance;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for calculating the Hausdorff similarity between two bags. The Hausdorff
 * similarity is the complement of the Hausdorff distance. The computation finally
 * relies in an instance distance function.
 *
 * @author Danel
 */
public class HausdorffSimilarity extends BagSimilarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Mathematical expression defining this function.
   */
  private Evaluable expression;

  /**
   * Creates a function which calculates the Hausdorff similarity between two
   * bags which is based on the Hausdorff distance.
   *
   * @param instanceDistanceClass class of the instance distance used by the
   * underlying Hausdorff distance.
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public HausdorffSimilarity(Class<? extends InstanceDistance> instanceDistanceClass,
          VarReader <Instance> X, VarReader <Instance> Y) {
    super(X, Y);
    expression = new Complement( new HausdorffDistance(instanceDistanceClass, X, Y));
  }

  /**
   * Creates a function which calculates the Hausdorff similarity between
   * two bags using the cosine distance between instances.
   *
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public HausdorffSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
    this(CosineDistance.class, X, Y);
  }

  /**
   * Evaluates the expression that defines the Hausdorff similarity in the
   * current values of the arguments.
   *
   * @return the value of the Hausdorff similarity.
   */
  @Override
  public double evaluate() {
    return expression.evaluate();
  }

}
