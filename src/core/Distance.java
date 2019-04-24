/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instance;

/**
 * Abstract class representing a normalized distance function, i.e., the values
 * of this function should be between 0 and 1.
 *
 * @author Danel
 */
public abstract class Distance extends Function {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function of two arguments <tt>Instance</tt>.
   */
  public Distance(VarReader <Instance> x, VarReader <Instance> y) {
    super(2);
    setArgument(1, x);
    setArgument(2, y);
  }

}
