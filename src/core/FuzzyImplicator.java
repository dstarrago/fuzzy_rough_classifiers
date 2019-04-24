/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Class for a fuzzy implication
 *
 * @author Danel
 */
public abstract class FuzzyImplicator extends BinaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a fuzzy implicator operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public FuzzyImplicator(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

}
