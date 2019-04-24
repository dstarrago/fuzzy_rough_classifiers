/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.owaWeighing;

import core.MaxWeighingMaker;
import java.io.Serializable;

/**
 * Class for generating weight vectors modelling the standard max operator.
 *
 * @author Danel
 */
public class StandardMaxWeighing extends MaxWeighingMaker implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  public double[] getWeightVector(int size) {
    double[] weights = new double[size];
    weights[0] = 1;
    return weights;
  }

}
