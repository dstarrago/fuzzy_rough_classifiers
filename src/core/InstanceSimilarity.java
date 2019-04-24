/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instance;

/**
 * Abstract class for similarity functions between instances.
 *
 * @author Danel
 */
public abstract class InstanceSimilarity extends Similarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  

  /**
   * Creates a function of two arguments <tt>Instance</tt>.
   */
  public InstanceSimilarity(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }

}
