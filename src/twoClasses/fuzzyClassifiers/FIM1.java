/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import core.iterators.InstancesFromBag;
import core.iterators.BagsFromClass;
import core.metrics.CosineSimilarity;
import core.multipleOperators.Max;
import core.multipleOperators.Average;
import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import core.*;

/**
 * Fuzzy instance based multi-instance classifiers using:
 * - membership function of bag to class: maximum
 * - membership function of instance to class: average to positive bags
 * - membership function of instance to bag: upper approximation
 * - Similarity between instances: cosine
 *
 * @author Danel
 */
public class FIM1 extends FMClassifier implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a FIM1 classifier by simply supplying its definition.
   */
  public FIM1() {
    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> B = new Var();   // a bag
    Var <Instance> x = new Var();   // an instance
    Var <Instance> y = new Var();   // another instance
    Var <Instances> BB = new Var(); // the training samples

    setMembership(new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X),
            new Average(new BagsFromClass(B, BB, CL),
            new Max(new InstancesFromBag(y, B),
            new CosineSimilarity(x, y))))));
  }

}
