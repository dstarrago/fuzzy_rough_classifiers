/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyRoughClassifiers;

import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 * Class equivalent to FRBM1 but implemented in the traditional way for comparison
 * purpose.
 *
 * Fuzzy-Rough bag based multi-instance classifiers using:
 * - membership function of bags to upper (lower) approximation of the class: standard max (min)
 * - Similarity between bags: Hausdorff similarity
 * - Similarity between instances: cosine similarity
 * - merging function of approximations membership degree: (default) weighted average
 *
 * @author Danel
 */
public class FRBMx1 extends AbstractClassifier implements Serializable {

  /**
   * Constant for the evaluation of the default merging function.
   */
  private double beta = 1;

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
      double s = 1 - cos(x, a);
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
  public double hausdorffDistance(Instance A, Instance B) {
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
  public double hausdorffSimilarity(Instance A, Instance B) {
    return 1 - hausdorffDistance(A, B);
  }

  /**
   * Computes the membership degree of a bag X to the lower approximation of
   * the given class.
   *
   * @param X the bag.
   * @param classLabel label of the given class.
   * @return the membership degree of a bag X to the lower approximation of
   * the given class.
   */
  private double lowerAppMembershipDegree(Instance X, int classLabel) {
    double min = 1;
    for (int j = 0; j < train.numInstances(); j++) {
      Instance B = train.instance(j);
      if (B.classValue() == classLabel) continue;  // BagsNotFromClass
      double R = hausdorffSimilarity(X, B);
      if (R < min)
        min = R;
    }
    return 1 - min;
  }

  /**
   * Computes the membership degree of a bag X to the upper approximation of
   * the given class.
   *
   * @param X the bag.
   * @param classLabel label of the given class.
   * @return the membership degree of a bag X to the upper approximation of
   * the given class.
   */
  private double upperAppMembershipDegree(Instance X, int classLabel) {
    double max = 0;
    for (int j = 0; j < train.numInstances(); j++) {
      Instance B = train.instance(j);
      if (B.classValue() != classLabel) continue;  // BagsFromClass
      double R = hausdorffSimilarity(X, B);
      if (R > max)
        max = R;
    }
    return max;
  }

  /**
   * Function used to merge upper and lower membership degrees into a single class
   * membership degree. The default merging function is the weighted average of
   * the membership degrees to the lower and upper class approximations. <tt>beta</tt>
   * is the weight of the lower approximation.
   */
  private double merge(double lowerApp, double upperApp) {
    return (beta * lowerApp + upperApp) / (beta + 1);
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
      double lowerAppMembDegree = lowerAppMembershipDegree(exmp, i);
      double upperAppMembDegree = upperAppMembershipDegree(exmp, i);

      // Calculates the final distribution
      distribution[i] = merge(lowerAppMembDegree, upperAppMembDegree);
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
