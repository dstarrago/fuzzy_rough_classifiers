/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Abstract class for functions. 
 * Functions encapsulates expression with one o more arguments.
 *
 * @author Danel
 */
public abstract class Function implements Evaluable, Serializable {

  /**
   * The arguments of the function
   */
  private VarReader [] arguments;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor creating the list of arguments of the function. Any class 
   * extending <tt>Function</tt> should call this constructor. 
   * 
   * @param numArguments the number of arguments of the function
   */
  public Function(int numArguments) {
    arguments = new VarReader[numArguments];
  }

  /**
   * Number of arguments of the function
   * @return the number of arguments
   */
  public int numArguments() {
    return arguments.length;
  }

  /**
   * Sets the value of one argument of the function.
   *
   * @param argIndex specifies which argument to update: 1 for the first argument,
   * 2 for the second and so on.
   * @param x the value that will be assigned to the argument.
   */
  public void setArgument(int argIndex, VarReader var) {
    arguments[argIndex - 1] = var;
  }

  /**
   * Return the value of one argument.
   * @param argIndex the index of the argument to return.
   * @return the value of the argument.
   */
  public VarReader argument(int argIndex) {
    return arguments[argIndex - 1];
  }

}
