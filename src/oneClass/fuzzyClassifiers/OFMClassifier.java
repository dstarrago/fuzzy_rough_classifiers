/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package oneClass.fuzzyClassifiers;

import core.MembershipToClass;

import java.io.Serializable;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.classifiers.Evaluation;

/**
 * Abstract class for One-class Fuzzy Multi-instance (OFM) classifiers.
 * One Class Fuzzy Multi-instance classifiers instead of having one
 * membership function for each class only has one membership function for the
 * positive class. A new bag is assigned to the positive class if its membership
 * degree to the positive class exceeds a certain threshold, otherwise it is
 * assigned to the negative class.
 *
 * @author Danel
 */
public abstract class OFMClassifier extends AbstractClassifier implements Serializable {

	
	
  /**
   * The membership function to the positive class
   */
  private MembershipToClass membership;

  /**
   * The training data
   */
  private Instances train;

  /**
   * Index of the positive class label. Default = 1.
   */
  private int posClass = 1;

  /**
   * Index of the negative class label. Default = 0.
   */
  private int negClass = 0;

  /**
   * Distance threshold. 
   */
  private double delta;

  /**
   * Increment used to find delta's optimal value.
   */
  private double stepWidth = 0.1;

  /**
   * Number of folds for internal cross validation in order to find the optimal
   * distance threshold. Default = 5.
   */
  private int folds = 5;

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

    // sets the training data as the samples for the membership function
    membership.sample().setObject(train);

    // Find the optimal threshold value
    optimalCVDelta();
    //optimalTrainDelta();
  }

  /**
   * Procedure to find the optimal value of the distance threshold
   * (parameter delta) using a cross validation scheme.
   */
  private void optimalCVDelta() {
    double avgDelta = 0;
    for (int i = 0; i < folds; i++) {
      Instances tr = train.trainCV(folds, i);
      Instances ts = train.testCV(folds, i);
      double d = findDelta(tr, ts);
      //System.out.println("Delta = " + d);
      avgDelta += d;
    }
    delta = avgDelta / folds;
    System.out.println("Optimal delta = " + delta);
  }
  
  /**
   * Procedure to find the optimal value of the distance threshold
   * (parameter delta) using the train set for testing.
   */
  private void optimalTrainDelta() {
    findDelta(train, train);
    System.out.println("Optimal delta = " + delta);
  }

  /**
   * Finds the optimal value of the distance threshold (parameter delta) on a given
   * train and test set.
   *
   * @param train de exemplars for training.
   * @param test de exemplars for testing.
   *
   * @return the optimal value of delta.
   */
  private double findDelta(Instances train, Instances test) {
    try {
      int t = 0;
      delta = 1;
      double opt = delta;
      double betterF1 = 0;
      int numSteps = (int)(1 / stepWidth + 1);
      double[] F1 = new double[numSteps];
      do {
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(this, test);
        //F1[t] = eval.fMeasure(posClass);
        F1[t] = eval.areaUnderROC(posClass);
        if (F1[t] > betterF1)
        {
          betterF1 = F1[t];
          opt = delta;
        }
        //System.out.println(String.format("Delta (%f) --> F1[%d] = %f", delta, t, F1[t]));
        if (t > 2 && F1[t-2] > F1[t-1] && F1[t-1] >= F1[t])
          break;
        delta -= stepWidth;
        t++;
      } while (delta > 0);
      delta = opt;
      //System.out.println("Optimal delta value: " + delta);
    } catch (Exception e) {
      System.out.println("Problems instantiating the class: " + e.getMessage());
    }
    return delta;
  }

  /**
   * Classifies the given test exemplar.
   *
   * @param exmp the exemplar to be classified.
   * @return the predicted most likely class for the exemplar.
   * @throws Exception if an error occurred during the prediction.
   */
  @Override
  public double classifyInstance(Instance exmp) throws Exception {
    // sets the instance for computing its membership degree
    membership.exemplar().setObject(exmp);

    // sets the target class to which the membership function belong
    membership.targetClass().setObject(posClass);

    // evaluates the membership funcion
    double posMembership = membership.evaluate();

    /**
     * Compare with the threshold and decide the class
     */
    return (posMembership >= delta)? posClass: negClass;
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
   * Sets the positive class label.
   * 
   * @param posClass the index of the positive class label.
   */
  public void setPosClass(int posClass) {
    this.posClass = posClass;
  }
  

}
