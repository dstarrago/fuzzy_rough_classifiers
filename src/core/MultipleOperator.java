/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Abstract class for an operator with multiple operands.
 *
 * @author Danel
 */
public abstract class MultipleOperator implements Evaluable, Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Placeholder for the expression that gives values to the operands
   */
  private Var<Evaluable> expression;
  //private Evaluable expression;

  /**
   * Controls the iterations of this multiple operator.
   */
  private Iterator iterator;

  /**
   * The operands after the function have been evaluated
   */
  private double[] operands;

  /**
   * Constructor for a multiple operator from an iterator and an explicit expression.
   * 
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public MultipleOperator(Iterator iterator, Evaluable expression) {
    this.expression = new Var();
    this.expression.setObject(expression);
    this.iterator = iterator;
  }

  /**
   * Constructor for a multiple operator from an iterator and an expression
   * placeholder.
   *
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public MultipleOperator(Iterator iterator, Var<Evaluable> expression) {
    this.expression = expression;
    this.iterator = iterator;
  }


  /**
   * Gets the expression that gives values to the operands.
   *
   * @return the Evaluable object representing the expression.
   */
  public Evaluable expression() {
    return expression.object();
  }

  /**
   * Gets access to the iterator.
   * 
   * @return the iterator.
   */
  public Iterator iterator() {
    return iterator;
  }

  /**
   * Number of operands.
   *
   * @return the number of operands.
   */
  public int numOperands() {
    if (iterator.numItems() == -1)
      throw new UnassignedOperandException("The number of operands is unknown!");
    return iterator.numItems();
  }

  /**
   * Gets the value of one of the operands.
   *
   * @param index the index of the operand which value is returned.
   * @return the value of the operand as double.
   */
  public double operand(int index) {
    if (operands == null)
      throw new UnassignedOperandException("No operands have been assigned");
    return operands[index];
  }

  /**
   * Gets the operands value.
   *
   * @return an array of double with each operand's value.
   */
  public double[] getOperands() {
    return operands;
  }

  /**
   * Evaluate the function in the dataset of instances
   * to gives value to the operands.
   */
  protected void evaluateOperands() {
    iterator.initialize();
    operands = new double[iterator.numItems()];
    int i = 0;
    while (iterator.hasNext()) {
      iterator.next();
      operands[i++] = expression.object().evaluate();
    }
  }

}
