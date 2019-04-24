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
 * Class equivalent to FBM1 but implemented in the traditional way for comparison
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
public class FBMx1 extends AbstractClassifier implements Serializable {

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
  private double cosSimilarity(Instance a, Instance b) {
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
   * Computes the cosine distance between two instances. Is the complement of
   * the cosine similarity.
   *
   * @param a the first instance.
   * @param b the second instance.
   * @return number between 0 and 1 representing the cosine distance.
   */
  private double cosDistance(Instance a, Instance b) {
    return 1 - cosSimilarity(a, b);
  }

  /**
   * Computes the minimal distance between an instances and a bag. It uses cosine
   * distance between instances.
   *
   * @param a the instance.
   * @param b the bag.
   * @return number between 0 and 1 representing the minimal distance.
   */
  private double min(Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      double s = cosDistance(x, a);
      //double s = NEuclidean(x, a);
      if (s < m)
        m = s;
    }
    return m;
  }

  /**
   * Computes the Hausdorff hemi-distance between two bags. It uses cosine
   * distance between instances.
   *
   * @param a the first bag.
   * @param b the second bag.
   * @return number between 0 and 1 representing the Hausdorff hemi-distance.
   */
  private double max_min(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      double s = min(x, B);
      if (s == 1) return 1;
      if (s > m)
        m = s;
    }
    return m;
  }

  /**
   * Computes the Hausdorff distance between two bags. It uses cosine
   * distance between instances.
   *
   * @param a the first bag.
   * @param b the second bag.
   * @return number between 0 and 1 representing the Hausdorff distance.
   */
  public double HausdorffDistance(Instance A, Instance B) {
    double n1 = max_min(A, B);
    if (n1 == 1) return 1;
    double n2 = max_min(B, A);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  /**
   * Computes the Hausdorff similarity between two bags. It uses cosine
   * distance between instances. Is the complement of the Hausdorff distance.
   *
   * @param a the first bag.
   * @param b the second bag.
   * @return number between 0 and 1 representing the Hausdorff distance.
   */
  public double HausdorffSimilarity(Instance A, Instance B) {
    return 1 - HausdorffDistance(A, B);
  }

  /**
   * Calculates the membership degree of instance x to the class with label 
   * <tt>classLabel</tt>.
   * 
   * @param x bag which membership degree has to be calculated.
   * @param classLabel index of the class.
   * @return
   */
  protected double calcMembership(Instance X, int classLabel) {
    double avg = 0;
    int count = 0;
    for (int j = 0; j < train.numInstances(); j++) {
      Instance B = train.instance(j);
      if (B.classValue() != classLabel) continue;
      avg += HausdorffSimilarity(X, B);
      count++;
    }
    avg /= count;
    return avg;
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
      // evaluates the membership funcion
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
