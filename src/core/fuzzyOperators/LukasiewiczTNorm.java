/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.fuzzyOperators;

import core.FuzzyTNorm;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the Lukasiewicz T-norm.
 *
 * @author Danel
 */
public class LukasiewiczTNorm extends FuzzyTNorm implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a Lukasiewicz t-norm operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public LukasiewiczTNorm(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

  /**
   * Evaluate the Lukasiewicz T-norm.
   *
   * @return the value of the Lukasiewicz T-norm over the operands.
   */
  public double evaluate() {
    double a = operand1().evaluate();
    double b = operand2().evaluate();
    double c = a + b - 1;
    return (c > 0)? c : 0;
  }

}
