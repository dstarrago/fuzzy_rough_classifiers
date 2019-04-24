/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * A holder for a constant double value.
 *
 * @author Danel
 */
public class Const implements VarReader <Double>, Evaluable, Serializable {

  /**
   * the contained double value.
   */
  private Double val;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates de constant object.
   *
   * @param val the double value to hold.
   */
  public Const(double val) {
    this.val = new Double(val);
  }

  /**
   * Gets the object Double it holds.
   *
   * @return the object Double it holds.
   */
  public Double object() {
    return val;
  }

  /**
   * Returns the value of the constant.
   *
   * @return the double value.
   */
  public double evaluate() {
    return val.doubleValue();
  }

}
