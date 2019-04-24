/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.owaWeighing;

import core.MinWeighingMaker;
import java.io.Serializable;

/**
 * Class for generating weight vectors modelling the standard min operator.
 *
 * @author Danel
 */
public class StandardMinWeighing extends MinWeighingMaker implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  public double[] getWeightVector(int size) {
    double[] weights = new double[size];
    weights[size - 1] = 1;
    return weights;
  }

}
