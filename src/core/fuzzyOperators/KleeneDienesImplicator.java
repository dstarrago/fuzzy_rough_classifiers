/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.fuzzyOperators;

import core.FuzzyImplicator;
import core.Evaluable;
import java.io.Serializable;

/**
 * Class for the Kleene-Dienes implicator.
 *
 * @author Danel
 */
public class KleeneDienesImplicator extends FuzzyImplicator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;
  
  /**
   * Create a Kleene-Dienes implicator operator.
   *
   * @param operand1 the first operand.
   * @param operand2 the second operand.
   */
  public KleeneDienesImplicator(Evaluable operand1, Evaluable operand2) {
    super(operand1, operand2);
  }

  /**
   * Evaluate the Kleene-Dienes implicator.
   *
   * @return the value of the Kleene-Dienes implicator over the operands.
   */
  public double evaluate() {
    double a = operand1().evaluate();
    double b = operand2().evaluate();
    double c = 1 - a;
    return (c > b)? c : b;
  }

}
