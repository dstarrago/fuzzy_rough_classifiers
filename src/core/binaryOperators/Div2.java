/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.binaryOperators;

import core.BinaryOperator;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the binary division operator.
 *
 * @author Danel
 */
public class Div2 extends BinaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  
  /**
   * Create a binary division operator.
   *
   * @param dividend the operand in the numerator.
   * @param divisor the operand in the denominator.
   */
  public Div2(Evaluable dividend, Evaluable divisor) {
    super(dividend, divisor);
  }

  /**
   * Evaluate the binary sum operator.
   *
   * @return the sum of the values of the two operands.
   *
   */
  public double evaluate() {
    double dividend = operand1().evaluate();
    double divisor = operand2().evaluate();
    return dividend / divisor;
  }

}
