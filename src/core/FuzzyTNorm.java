/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Class for a fuzzy T-norm
 *
 * @author Danel
 */
public abstract class FuzzyTNorm extends BinaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a fuzzy t-norm operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public FuzzyTNorm(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

}
