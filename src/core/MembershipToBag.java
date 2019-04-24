/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;
import weka.core.Instance;

/**
 * Class defining the membership function of instances to a bag.
 *
 * @author Danel
 */
public class MembershipToBag extends ModeledFunction implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for the class membership.
   * @param exemplar variable refering to the instance to which the membership
   * function will be calculated.
   * @param bag the bag this object refers to.
   * @param expression mathematical expression this function represents.
   */
  public MembershipToBag(Var<Instance> exemplar, Var<Instance> bag,
          Evaluable expression) {
    super(expression, 2);
    setArgument(1, exemplar);
    setArgument(2, bag);
  }

  /**
   * Gets access to the instance which membership function will be calculated.
   *
   * @return the variable holding the instance which membership function will be
   * calculated.
   */
  public Var<Instance> exemplar() {
    return (Var<Instance>)argument(1);
  }

  /**
   * Gets access to the bag this object refers to.
   * @return the variable holding the bag this object refers to.
   */
  public Var<Instance> bag() {
    return (Var<Instance>)argument(2);
  }

}
