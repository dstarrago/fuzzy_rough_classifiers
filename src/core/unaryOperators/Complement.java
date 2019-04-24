/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.unaryOperators;

import core.Evaluable;
import core.UnaryOperator;
import java.io.Serializable;

/**
 * Operator that returns one minus the operand's value.
 *
 * @author Danel
 */
public class Complement extends UnaryOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  
  /**
   * Create a complement for an operand.
   *
   * @param operand the operand.
   */
  public Complement(Evaluable operand) {
    super(operand);
  }

  /**
   * Evaluates the complement operator.
   *
   * @return the complement of the operand's value, i.e., one minus the operand's value.
   */
  public double evaluate() {
    return 1 - operand().evaluate();
  }

}
