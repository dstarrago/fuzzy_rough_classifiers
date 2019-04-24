/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyRoughClassifiers;

import core.iterators.InstancesFromBag;
import core.iterators.BagsNotFromClass;
import core.iterators.BagsFromClass;
import core.metrics.CosineSimilarity;
import core.multipleOperators.Max;
import core.multipleOperators.Min;
import core.multipleOperators.Average;
import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import core.*;
import core.fuzzyOperators.LukasiewiczImplicator;
import core.fuzzyOperators.LukasiewiczTNorm;

/**
 * Fuzzy-Rough instance based multi-instance classifiers using:
 * - membership function of bags to upper and lower approximations of the class: maximum
 * - membership function of instance to upper (lower) approximation of the class: standard max (min)
 * - Similarity between instances: cosine
 * - membership function of instance to class: average to positive bags
 * - membership function of instance to bag: upper approximation
 * - implicator: Lukasiewicz
 * - T-norm: Lukasiewicz
 * - merging function of approximations membership degree: (default) weighted average
 *
 * @author Danel
 */
public class FRIM1 extends FRMClassifier implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a FRIM1 classifier by simply supplying its definition.
   */
  public FRIM1() {
    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> B = new Var();   // a bag
    Var <Instance> A = new Var();   // another bag
    Var <Instance> x = new Var();   // an instance
    Var <Instance> y = new Var();   // another instance
    Var <Instance> z = new Var();   // another instance
    Var <Instances> BB = new Var(); // the training samples

    ModeledFunction instanceToClassMembership =
            new MembershipToClass(y, BB, CL,
            new Average(new BagsFromClass(A, BB, CL),
            new Max(new InstancesFromBag(z, A),
            new CosineSimilarity(y, z))));

    setLowerAppMembership(new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X),
            new Min(new BagsNotFromClass(B, BB, CL),
            new Min(new InstancesFromBag(y, B),
            new LukasiewiczImplicator(
            new CosineSimilarity(x, y), instanceToClassMembership))))));

    setUpperAppMembership(new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X),
            new Max(new BagsFromClass(B, BB, CL),
            new Max(new InstancesFromBag(y, B),
            new LukasiewiczTNorm(
            new CosineSimilarity(x, y), instanceToClassMembership))))));
  }

}
