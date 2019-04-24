/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.binaryOperators;

import core.BinaryOperator;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the binary max operator.
 *
 * @author Danel
 */
public class Max2 extends BinaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  
  /**
   * Create a binary max operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public Max2(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

  /**
   * Evaluate the binary max operator.
   *
   * @return the larger of the values of the two operands.
   *
   */
  public double evaluate() {
    double a = operand1().evaluate();
    double b = operand2().evaluate();
    return (a > b)? a : b;
  }

}
