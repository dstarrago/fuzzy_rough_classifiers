/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.owaWeighing;

import core.MinWeighingMaker;
import java.io.Serializable;

/**
 * Class for generating lineal ascending weight vectors.
 *
 * @author Danel
 */
public class LinealMinWeighing extends MinWeighingMaker implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  public double[] getWeightVector(int size) {
    double[] weights = new double[size];
    double p = size;
    double Z = p * (p + 1);
    for (int i = 1; i <= size; i++) {
      weights[i - 1] = 2 * i / Z;
    }
    return weights;
  }

}
