/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Abstract class for objects that generates weight vectors.
 *
 * @author Danel
 */
public abstract class WeighingMaker implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  abstract public double[] getWeightVector(int size);

}
