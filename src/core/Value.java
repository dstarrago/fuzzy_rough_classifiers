/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * A holder for a variable double value.
 *
 * @author Danel
 */
public class Value implements VarWriter <Double>, Evaluable, Serializable {

  /**
   * the contained double value.
   */
  private Double val;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Sets the object Double to hold.
   *
   * @param object the object Double to hold.
   */
  public void setObject(Double val) {
    this.val = val;
  }
  
  /**
   * Creates a new Double to hold from a provided double value.
   *
   * @param object the double value to hold.
   */
  public void setValue(double val) {
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
   * Returns the value of the variable.
   *
   * @return the double value.
   */
  public double evaluate() {
    return val.doubleValue();
  }

}
