/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instance;

/**
 * Class for the function that returns the size of a bag.
 *
 * @author Danel
 */
public class SizeOfBag extends Function {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function of one argument <tt>Instance</tt>.
   */
  public SizeOfBag(VarReader <Instance> X) {
    super(1);
    setArgument(1, X);
  }

  @Override
  public double evaluate() {
    Instance X = (Instance)argument(1).object();
    return X.relationalValue(1).numInstances();
  }

}
