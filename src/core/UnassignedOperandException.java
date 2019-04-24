/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * Exception that is raised when trying to evaluate something that has no
 * reference to some required operand.
 *
 * @author Danel
 */
public class UnassignedOperandException  extends RuntimeException
  implements Serializable{

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a new UnassignedOperandException with no message.
   *
   */
  public UnassignedOperandException() {

    super();
  }

  /**
   * Creates a new UnassignedOperandException.
   *
   * @param message the reason for raising an exception.
   */
  public UnassignedOperandException(String message) {

    super(message);
  }

}
