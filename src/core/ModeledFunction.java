/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

/**
 * Class for functions (subclass of Function) which can be described
 * with an evaluable expression.
 *
 * @author Danel
 */
public class ModeledFunction extends Function {

  /**
   * Mathematical expression this function represents
   */
  private Evaluable expression;

  /**
   * Constructor for the <tt>ModeledFunction</tt>. 
   * 
   * @param expression the mathematical expression of the function.
   * @param numArguments the number of arguments of the function
   */
  public ModeledFunction(Evaluable expression, int numArguments) {
    super(numArguments);
    this.expression = expression;
  }

  /**
   * Gets the mathematical expression of the function.
   * @return the <tt>Evaluable</tt> object representing the expression.
   */
  public Evaluable expression() {
    return expression;
  }

  /**
   * Evaluates the expression based on the argument's values.
   *
   * @return the value of the expression.
   */
  public double evaluate() {
    return expression.evaluate();
  }

}
