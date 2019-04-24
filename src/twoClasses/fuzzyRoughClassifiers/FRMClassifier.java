/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyRoughClassifiers;

import core.binaryOperators.Div2;
import core.binaryOperators.Mult2;
import core.binaryOperators.Sum2;
import core.*;
import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Abstract class for Fuzzy-Rough Multi-instance (FRM) classifiers. Fuzzy-Rough
 * Multi-instance classifiers are defined by membership functions of bags to
 * both the class upper and lower approximations. The membership function can be
 * instance-based or bag-based. Use <tt>setLowerAppMembership(MembershipToClass f)
 * </tt> and <tt>setUpperAppMembership(MembershipToClass f)</tt> to set the lower
 * and upper membership function respectively.
 * <p>
 * You can extend this class or instantiate it and assign a membership function
 * for classification purpose.
 *
 * @author Danel
 */
public class FRMClassifier extends AbstractClassifier implements Serializable {

  /**
   * The name of this classifier;
   */
  private String name;

  /**
   * The membership function to upper approximation class which characterize this classifier.
   */
  private MembershipToClass upperAppMembership;

  /**
   * Membership degree to the upper approximation of the class.
   */
  private Value upperAppMembDegree = new Value();

  /**
   * The membership function to lower approximation class which characterize this classifier.
   */
  private MembershipToClass lowerAppMembership;

  /**
   * Membership degree to the lower approximation of the class.
   */
  private Value lowerAppMembDegree = new Value();

  /**
   * Constant for the evaluation of the default merging function.
   */
  private Const beta = new Const(1);

  /**
   * Constant = 1.0 for the evaluation of the default merging function.
   */
  private final Const one = new Const(1);

  /**
   * Function used to merge upper and lower membership degrees into a single class
   * membership degree. The default merging function is the weighted average of
   * the membership degrees to the lower and upper class approximations. <tt>beta</tt>
   * is the weight of the lower approximation.
   */
  private Evaluable merge = new Div2(new Sum2(new Mult2(beta, lowerAppMembDegree), upperAppMembDegree),
                                     new Sum2(beta, one));

  /**
   * The training data
   */
  private Instances train;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Builds the classifier
   *
   * @param train the training data
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    train = new Instances(data);
    train.deleteWithMissingClass();

    // sets the training data as the samples for both membership functions
    upperAppMembership.sample().setObject(train);
    lowerAppMembership.sample().setObject(train);
  }

  /**
   * Computes the distribution for a given exemplar
   *
   * @param exmp the exemplar for which distribution is computed
   * @return the distribution
   * @throws Exception if the distribution can't be computed successfully
   */
  @Override
  public double[] distributionForInstance(Instance exmp) throws Exception {
    int numClasses = exmp.dataset().classAttribute().numValues();
    double [] distribution = new double[numClasses];

    // sets the instance for computing its membership degree
    upperAppMembership.exemplar().setObject(exmp);
    lowerAppMembership.exemplar().setObject(exmp);

    for (int i = 0; i < numClasses; i++) {

      // sets the target class to which the membership function belong
      upperAppMembership.targetClass().setObject(i);
      lowerAppMembership.targetClass().setObject(i);

      // evaluates the membership funcion
      lowerAppMembDegree.setValue(lowerAppMembership.evaluate());
      upperAppMembDegree.setValue(upperAppMembership.evaluate());

      // Calculates the final distribution
      distribution[i] = merge.evaluate();
    }
    return distribution;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.RELATIONAL_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // other
    result.enable(Capability.ONLY_MULTIINSTANCE);

    return result;
  }

  /**
   * Sets the membership function for the upper aproximation of classes.
   * 
   * @param f the bag-to-class membership function.
   */
  public void setUpperAppMembership(MembershipToClass f) {
    upperAppMembership = f;
  }

  /**
   * Sets the membership function for the lower aproximation of classes.
   *
   * @param f the bag-to-class membership function.
   */
  public void setLowerAppMembership(MembershipToClass f) {
    lowerAppMembership = f;
  }

  /**
   * Sets the mathmatical expresion defining the function used to merge upper
   * and lower membership degrees into a single class membership degree.
   *
   * @param expression a valid evaluable expression.
   */
  public void setMergeFunction(Evaluable expression) {
    merge = expression;
  }

  /**
   * Sets the constant used in the evaluation of the default merging function.
   * The default merging function is the weighted average of the membership degrees
   * to the lower and upper class approximations. <tt>beta</tt> is the weight of
   * the lower approximation.
   *
   * The value of <tt>beta</tt> indicates how many times the lower approximation
   * is more important than the upper aproximation in the final calculus of the
   * membership degree of the bag to the class.
   *
   * @param val the new double value of <tt>beta</tt>.
   */
  public void setBeta(double val) {
    beta = new Const(val);
  }

  /**
   * Gets the name of this classifier.
   */
  public String name() {
    return name;
  }

  /**
   * Sets the name of this classifier.
   *
   * @param name string that has to identify this classifier.
   */
  public void setName(String name) {
    this.name = name;
  }

}
