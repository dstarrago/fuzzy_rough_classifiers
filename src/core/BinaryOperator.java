/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Abstract class for an operator with two operands.
 *
 * @author Danel
 */
public abstract class BinaryOperator implements Evaluable, Serializable {

  /**
   * The first operand
   */
  private Evaluable operand1;

  /**
   * The second operand
   */
  private Evaluable operand2;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a binary operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public BinaryOperator(Evaluable operand1, Evaluable operand2) {
    this.operand1 = operand1;
    this.operand2 = operand2;
  }

  /**
   * Gets the first operand's object.
   *
   * @return the first operand.
   */
  public Evaluable operand1() {
    if (operand1 == null)
      throw new UnassignedOperandException("The operand have not been assigned");
    return operand1;
  }

  /**
   * Gets the second operand's object.
   *
   * @return the second operand.
   */
  public Evaluable operand2() {
    if (operand2 == null)
      throw new UnassignedOperandException("The operand have not been assigned");
    return operand2;
  }

}
