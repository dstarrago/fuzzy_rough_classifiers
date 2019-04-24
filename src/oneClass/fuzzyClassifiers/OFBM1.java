/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package oneClass.fuzzyClassifiers;

import core.iterators.BagsFromClass;
import core.metrics.HausdorffSimilarity;
import core.multipleOperators.Average;
import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import core.*;

/**
 * One-class Fuzzy bag based multi-instance (OFBM) classifiers using:
 * - membership function of bag to class: average
 * - Similarity between bags: Hausdorff
 * - Similarity between instances: cosine
 *
 * @author Danel
 */
public class OFBM1 extends OFMClassifier implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  public OFBM1() {
    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> B = new Var();   // a bag
    Var <Instances> BB = new Var(); // the training samples

    setMembership(new MembershipToClass(X, BB, CL,
            new Average(new BagsFromClass(B, BB, CL),
            new HausdorffSimilarity(X, B))));
  }

}
