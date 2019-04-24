/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import core.MembershipToClass;
import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Basic class for Fuzzy Multi-instance (FM) classifiers. Fuzzy Multi-instance
 * classifiers are defined by a membership function of bags to the class. The
 * membership function can be instance-based or bag-based. Use <tt>setMembership
 * (MembershipToClass f)</tt> to set the membership function.
 * <p>
 * You can extend this class or instantiate it and assign a membership function
 * for classification purpose.
 *
 * @author Danel
 */
public class FMClassifier extends AbstractClassifier implements Serializable {

  /**
   * The name of this classifier;
   */
  private String name;

  /**
   * The membership function to the classes which characterize this classifier
   */
  private MembershipToClass membership;

  /**
   * The training data
   */
  private Instances train;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Builds the classifier
   *
   * @param data the training data
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    train = new Instances(data);
    train.deleteWithMissingClass();

    // sets the training data as the samples for the membership function
    membership.sample().setObject(train);
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
    membership.exemplar().setObject(exmp);

    for (int i = 0; i < numClasses; i++) {

      // sets the target class to which the membership function belong
      membership.targetClass().setObject(i);

     
      // evaluates the membership funcion
      //System.out.println("Instance of class " + exmp.classIndex() + ", " + exmp.classValue());
      distribution[i] = membership.evaluate();
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
   * Sets the membership function of this fuzzy classifier.
   * 
   * @param f the bag-to-class membership function.
   */
  public void setMembership(MembershipToClass f) {
    membership = f;
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

