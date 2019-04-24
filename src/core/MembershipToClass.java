/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class defining the membership function of exemplars to a class. Exemplars can
 * be instances or bags.
 *
 * @author Danel
 */
public class MembershipToClass extends ModeledFunction implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for the class membership.
   * @param exemplar variable refering to the sample to which the membership
   * function will be calculated.
   * @param sample Set of samples representing the class this object refers to.
   * @param targetClass the class this object refers to.
   * @param expression mathematical expression this function represents.
   */
  public MembershipToClass(Var<Instance> exemplar, Var<Instances> sample,
          Var<Integer> targetClass, Evaluable expression) {
    super(expression, 3);
    setArgument(1, exemplar);
    setArgument(2, sample);
    setArgument(3, targetClass);
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
   * Gets access to the set of samples representing the class this object refers to.
   * @return the variable holding the set of samples representing the class
   * this object refers to.
   */
  public Var<Instances> sample() {
    return (Var<Instances>)argument(2);
  }

  /**
   * Gets access to the class this object refers to.
   *
   * @return the variable holding the index of the class this object refers to.
   */
  public Var<Integer> targetClass() {
    return (Var<Integer>)argument(3);
  }

}
