/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;

/**
 * A holder for an object type variable.
 *
 * @author Danel
 */
public class Var <T> implements VarWriter <T>, Serializable {

  /**
   * the contained object.
   */
  private T object;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Sets the object to hold.
   *
   * @param object the object to hold.
   */
  public void setObject(T object) {
    this.object = object;
  }

  /**
   * Gets the object it holds.
   *
   * @return the object it holds.
   */
  public T object() {
    return object;
  }

}
