/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package oneClass.fuzzyRoughClassifiers;

import core.binaryOperators.Div2;
import core.binaryOperators.Mult2;
import core.binaryOperators.Sum2;
import core.*;
import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Abstract class for One-class Fuzzy-Rough Multi-instance (OFRM) classifiers.
 * One Class Fuzzy-Rough Multi-instance classifiers instead of having one
 * membership function for the approximations of each class only has one membership
 * function for the approximations of the positive class. A new bag is assigned
 * to the positive class if the aggregation of its membership degree to the
 * lower and upper aproximations to the positive class exceeds a certain threshold,
 * otherwise it is assigned to the negative class.
 * @author Danel
 */
public abstract class OFRMClassifier extends AbstractClassifier implements Serializable {

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

    // sets the training data as the samples for both membership functions
    upperAppMembership.sample().setObject(train);
    lowerAppMembership.sample().setObject(train);

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
    upperAppMembership.exemplar().setObject(exmp);
    lowerAppMembership.exemplar().setObject(exmp);

    // sets the target class to which the membership function belong
    upperAppMembership.targetClass().setObject(posClass);
    lowerAppMembership.targetClass().setObject(posClass);

    // evaluates the membership funcion to the approximations
    lowerAppMembDegree.setValue(lowerAppMembership.evaluate());
    upperAppMembDegree.setValue(upperAppMembership.evaluate());

    // Calculates the final distribution
    double posMembership = merge.evaluate();

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
   * Sets the positive class label.
   *
   * @param posClass the index of the positive class label.
   */
  public void setPosClass(int posClass) {
    this.posClass = posClass;
  }

}
