/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Abstract class for an operator with a single operand.
 *
 * @author Danel
 */
public abstract class UnaryOperator implements Evaluable, Serializable {

  /**
   * The operand
   */
  private Evaluable operand;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Create a unary operator.
   *
   * @param operand the operand.
   */
  public UnaryOperator(Evaluable operand) {
    this.operand = operand;
  }

  /**
   * Gets the operand's object.
   *
   * @return the operand.
   */
  public Evaluable operand() {
    if (operand == null)
      throw new UnassignedOperandException("The operand have not been assigned");
    return operand;
  }

}
