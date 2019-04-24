/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instance;

/**
 * Abstract class for distance functions between instances.
 *
 * @author Danel
 */
public abstract class InstanceDistance extends Distance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function of two arguments <tt>Instance</tt>.
   */
  public InstanceDistance(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }

}
