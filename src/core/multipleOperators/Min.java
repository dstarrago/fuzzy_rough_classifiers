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
 * Class for an operator that return the minimum value of its operands.
 *
 * @author Danel
 */
public class Min extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for multiple Min operator.
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public Min(Iterator iterator, Evaluable expression) {
    super(iterator, expression);
  }

  /**
   * Constructor for multiple Min operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public Min(Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
  }

  /**
   * Evaluate the minimum operator.
   *
   * @return the minimum value of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double min = Double.POSITIVE_INFINITY;
    for (int i = 0; i < numOperands(); i++) {
      double actualVal = operand(i);
      if (actualVal < min)
        min = actualVal;
    }
    return min;
  }

}
