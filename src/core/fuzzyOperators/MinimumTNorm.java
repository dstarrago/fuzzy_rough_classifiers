/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.fuzzyOperators;

import core.FuzzyTNorm;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the Minimum T-norm.
 *
 * @author Danel
 */
public class MinimumTNorm extends FuzzyTNorm implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a Minimum TNorm operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public MinimumTNorm(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

  /**
   * Evaluate the Minimum T-norm.
   *
   * @return the value of the Minimum T-norm over the operands.
   */
  public double evaluate() {
    double a = operand1().evaluate();
    double b = operand2().evaluate();
    return (a < b)? a: b;
  }

}
