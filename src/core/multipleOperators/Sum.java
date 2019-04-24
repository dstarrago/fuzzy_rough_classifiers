/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.multipleOperators;

import core.Evaluable;
import core.Iterator;
import core.MultipleOperator;
import core.Var;
import java.io.Serializable;

/**
 * Class for an operator that return the sum of values of its operands.
 *
 * @author Danel
 */
public class Sum extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for multiple Sum operator.
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public Sum(Iterator iterator, Evaluable expression) {
    super(iterator, expression);
  }

  /**
   * Constructor for multiple Sum operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public Sum(Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
  }

  /**
   * Evaluate the sum operator.
   *
   * @return the sum of values of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double sum = 0;
    for (int i = 0; i < numOperands(); i++) {
      sum += operand(i);
    }
    return sum;
  }

}
