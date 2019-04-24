/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instance;

/**
 * Abstract class for similarity functions between bags.
 *
 * @author Danel
 */
public abstract class BagSimilarity extends Similarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function of two arguments <tt>Instance</tt> which represent bags.
   */
  public BagSimilarity(VarReader <Instance> X, VarReader <Instance> Y) {
    super(X, Y);
  }

}
