/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.binaryOperators;

import core.BinaryOperator;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the binary multiplication operator.
 *
 * @author Danel
 */
public class Mult2 extends BinaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  
  /**
   * Create a binary multiplication operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public Mult2(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

  /**
   * Evaluate the binary multiplication operator.
   *
   * @return the multiplication of the values of the two operands.
   *
   */
  public double evaluate() {
    double a = operand1().evaluate();
    double b = operand2().evaluate();
    return a * b;
  }

}
