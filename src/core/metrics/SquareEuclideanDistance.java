/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.InstanceDistance;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for the Square Euclidean Distance between instances. 
 *
 * @author Danel
 */
public class SquareEuclideanDistance extends InstanceDistance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function which calculates the Square Euclidean Distance between two instances.
   *
   * @param x variable for the first instance.
   * @param y variable for the second instance.
   */
  public SquareEuclideanDistance(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }
  
  private double square_euclidean(Instance a, Instance b) {
    double s = 0;
    double dif;
    for (int i = 0; i < a.numAttributes(); i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
    }
    return s;
  }

  @Override
  public double evaluate() {
    Instance a = (Instance)argument(1).object();
    Instance b = (Instance)argument(2).object();
    return square_euclidean(a, b);
  }


}
