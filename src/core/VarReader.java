/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

/**
 * Variable holder which content can be read.
 *
 * @author Danel
 */
public interface VarReader <T> {

  /**
   * Gets the object it holds.
   * 
   * @return the object it holds.
   */
  T object();

}
