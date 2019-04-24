/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.multipleOperators;

import core.Evaluable;
import core.Iterator;
import core.MembershipToClass;
import core.MultipleOperator;
import core.Var;
import core.iterators.InstancesFromBag;
import java.io.Serializable;

/**
 * Class for an operator that return the maximum value of its operands.
 *
 * @author Danel
 */
public class Max extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for multiple Max operator.
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public Max(Iterator iterator, Evaluable expression) {
    super(iterator, expression);
  }

  /**
   * Constructor for multiple Max operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public Max(Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
  }

  /**
   * Evaluate the maximum operator.
   *
   * @return the maximum value of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < numOperands(); i++) {
      double actualVal = operand(i);
      if (actualVal > max)
        max = actualVal;
    }
    return max;
  }

}
