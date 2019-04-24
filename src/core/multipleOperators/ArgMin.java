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
 * Class for an operator that return the index of the minimum value of its operands.
 *
 * @author Danel
 */
public class ArgMin extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for multiple ArgMin operator from an iterator and an explicit
   * expression.
   *
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public ArgMin(Iterator iterator, Evaluable expression) {
    super(iterator, expression);
  }

  /**
   * Constructor for multiple ArgMin operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public ArgMin(Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
  }

  /**
   * Evaluate the argument of the minimum operator.
   *
   * @return the index of the minimum value of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double min = Double.POSITIVE_INFINITY;
    double minIndex = 0;
    for (int i = 0; i < numOperands(); i++) {
      double actualVal = operand(i);
      if (actualVal < min) {
        min = actualVal;
        minIndex = i;
      }
    }
    return minIndex;
  }

}
