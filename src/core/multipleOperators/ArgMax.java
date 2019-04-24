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
 * Class for an operator that return the index of the maximum value of its operands.
 *
 * @author Danel
 */
public class ArgMax extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for multiple ArgMax operator from an iterator and an explicit
   * expression.
   *
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public ArgMax(Iterator iterator, Evaluable expression) {
    super(iterator, expression);
  }

  /**
   * Constructor for multiple ArgMax operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public ArgMax(Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
  }

  /**
   * Evaluate the argument of the maximum operator.
   *
   * @return the index of the maximum value of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double max = Double.NEGATIVE_INFINITY;
    double maxIndex = 0;
    for (int i = 0; i < numOperands(); i++) {
      double actualVal = operand(i);
      if (actualVal > max) {
        max = actualVal;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

}
