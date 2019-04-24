/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import core.iterators.BagsFromClass;
import core.metrics.EuclideanDistance;
import core.metrics.HausdorffSimilarity;
import core.multipleOperators.Average;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import core.*;

/**
 * Fuzzy bag based multi-instance classifiers using:
 * - membership function of bag to class: average
 * - Similarity between bags: Hausdorff
 * - Similarity between instances: cosine
 *
 * @author Danel
 */
public class FBM1 extends FMClassifier implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a FBM1 classifier by simply supplying its definition.
   */
  public FBM1() {
    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> B = new Var();   // a bag
    Var <Instances> BB = new Var(); // the training samples

    setMembership(new MembershipToClass(X, BB, CL,
            new Average(new BagsFromClass(B, BB, CL),
            //new HausdorffSimilarity(X, B))));
    		new HausdorffSimilarity(EuclideanDistance.class, X, B))));
  }

}
