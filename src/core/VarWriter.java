/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

/**
 * Variable holder which content can be read and write.
 *
 * @author Danel
 */
public interface VarWriter <T> extends VarReader <T> {

  /**
   * Sets the object to hold.
   * 
   * @param object the object to hold.
   */
  void setObject(T object);

}
