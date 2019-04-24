/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.InstanceDistance;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for the Normalized Euclidean Distance between instances. 
 * The Normalized Euclidean Distance is between 0 and 1.
 *
 * @author Danel
 */
public class NormalizedEuclideanDistance extends InstanceDistance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function which calculates the Euclidean Distance between two instances.
   *
   * @param x variable for the first instance.
   * @param y variable for the second instance.
   */
  public NormalizedEuclideanDistance(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }
  
  private double normalized_euclidean(Instance a, Instance b) {
    double s = 0;
    double dif, a2 = 0, b2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    double D = Math.sqrt(a2) + Math.sqrt(b2);
    return Math.sqrt(s) / D;
  }

  @Override
  public double evaluate() {
    Instance a = (Instance)argument(1).object();
    Instance b = (Instance)argument(2).object();
    return normalized_euclidean(a, b);
  }


}
