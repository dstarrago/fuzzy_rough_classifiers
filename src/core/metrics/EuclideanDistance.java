/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.InstanceDistance;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for the Euclidean Distance between instances. 
 *
 * @author Danel
 */
public class EuclideanDistance extends InstanceDistance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function which calculates the Euclidean Distance between two instances.
   *
   * @param x variable for the first instance.
   * @param y variable for the second instance.
   */
  public EuclideanDistance(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }
  
  private double euclidean(Instance a, Instance b) {
    double s = 0;
    double dif;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
    }
    return Math.sqrt(s);
  }

  @Override
  public double evaluate() {
    Instance a = (Instance)argument(1).object();
    Instance b = (Instance)argument(2).object();
    return euclidean(a, b);
  }


}
