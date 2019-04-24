/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Class equivalent to FIM1 but implemented in the traditional way for comparison
 * purpose.
 *
 * Fuzzy instance based multi-instance classifiers using:
 * - membership function of bag to class: maximum
 * - membership function of instance to class: average to positive bags
 * - membership function of instance to bag: upper approximation
 * - Similarity between instances: cosine
 *
 * @author Danel
 */
public class FIMx1 extends AbstractClassifier implements Serializable {

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
  }

  /**
   * Computes the cosine similarity between two instances.
   *
   * @param a the first instance in the comparison.
   * @param b the second instance in the comparison.
   * @return number between 0 and 1 representing the cosine similarity.
   */
  private double cos(Instance a, Instance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numAttributes(); i++) {
      ab += a.value(i) * b.value(i);
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    if (ab == 0) return 0;   // (a2 == 0) or (b2 == 0) => (ab == 0)
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));
    return (s > 1)? 1 : s;
  }

  /**
   * Calculates the membership degree of instance x to the class with label 
   * <tt>classLabel</tt>.
   * 
   * @param x bag which membership degree has to be calculated.
   * @param classLabel index of the class.
   * @return the class membership.
   */
  protected double calcMembership(Instance X, int classLabel) {
    double max = 0;
    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
      Instance x = X.relationalValue(1).instance(i);
      double avg = 0;
      int count = 0;
      for (int j = 0; j < train.numInstances(); j++) {
        Instance B = train.instance(j);
        if (B.classValue() != classLabel) continue;
        double upper = 0;
        for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
          Instance y = B.relationalValue(1).instance(k);
          double R = cos(x, y);
          if (R > upper)
            upper = R;
        }
        avg += upper;
        count++;
      }
      avg /= count;
      if (avg > max)
        max = avg;
    }
    return max;
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

    for (int i = 0; i < numClasses; i++) {
      // evaluates the membership function
      distribution[i] = calcMembership(exmp, i);
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

}
