/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package oneClass.fuzzyRoughClassifiers;

import core.unaryOperators.Complement;
import core.iterators.BagsNotFromClass;
import core.iterators.BagsFromClass;
import core.metrics.HausdorffSimilarity;
import core.multipleOperators.Max;
import core.multipleOperators.Min;
import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import core.*;

/**
 * One-class Fuzzy-Rough Bag based Multi-instance (OFRBM) classifiers using:
 * - membership function of bags to upper (lower) approximation of the class: standard max (min)
 * - Similarity between bags: Hausdorff similarity
 * - Similarity between instances: cosine similarity
 * - merging function of approximations membership degree: (default) weighted average
 *
 * @author Danel
 */
public class OFRBM1 extends OFRMClassifier implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  public OFRBM1() {
    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> B = new Var();   // a bag
    Var <Instances> BB = new Var(); // the training samples

    setLowerAppMembership(new MembershipToClass(X, BB, CL,
            new Complement(
            new Min(new BagsNotFromClass(B, BB, CL),
            new HausdorffSimilarity(X, B)))));

    setUpperAppMembership(new MembershipToClass(X, BB, CL,
            new Max(new BagsFromClass(B, BB, CL),
            new HausdorffSimilarity(X, B))));
  }

}
