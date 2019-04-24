/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyRoughClassifiers;

import core.multipleOperators.*;
import core.iterators.*;
import core.Evaluable;
import core.Var;
import core.metrics.CosineSimilarity;
import core.MembershipToClass;
import core.fuzzyOperators.LukasiewiczImplicator;
import core.fuzzyOperators.LukasiewiczTNorm;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Two-class Fuzzy-Rough Instance-based Multi-instance (TFRIM) classifier family
 * factory. Class for generating any classifier from the TFRIM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one TFRIM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the TFRIM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * TFRIMFactory f = new TFRIMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FRMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Danel
 */
public class TFRIMFactory {

  /**
   * Number of definitions of bag-to-class-approximation membership functions
   */
  public static final int numBagToClassAppMembership = 3;

  /**
   * Number of definitions of instance-to-class-approximation membership functions
   */
  public static final int numInstanceToClassAppMembership = 2;

  /**
   * Number of definitions of instance-to-class membership functions
   */
  public static final int numInstanceToClassMembership = 4;

  /**
   * Number of definitions of instance-to-bag membership functions
   */
  public static final int numInstanceToBagMembership = 3;

  /**
   * Number of combinations of the last two terms in the family membership
   * function definition.
   */
  private int interSize1 = numInstanceToBagMembership * numInstanceToClassMembership;

  /**
   * Number of combinations of the last three terms in the family membership
   * function definition.
   */
  private int interSize2 = interSize1 * numInstanceToClassAppMembership;

  /**
   * Size of the TFRIM family, based on the declared number of definitions.
   */
  private int size = interSize2 * numBagToClassAppMembership;

  /**
   * For composing the name of the generated classifier.
   */
  private String name;

  /**
   * Flag to avoid duplicates in the composition of the classifier name
   */
  private boolean allowSignature;

  /**
   * Creates a bag-to-class-lower-approximation membership function corresponding
   * to the <tt>index</tt>-th bag-to-class-lower-approximation membership
   * definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a bag-to-class-lower-approximation membership function.
   */
  private MembershipToClass getBagToClassLowerAppMembership(int index) {

    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> x = new Var();   // an instance
    Var <Instances> BB = new Var(); // the training samples

    MembershipToClass M = null;
    int bagToClassIndex = index / interSize2;

    switch (bagToClassIndex) {
      case 0:
        if (allowSignature) name += "-Max";
        M = new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X), getInstanceToClassLowerAppMembership(index, x, BB, CL)));
        break;
      case 1:
        if (allowSignature) name += "-OWAmax";
        M = new MembershipToClass(X, BB, CL,
            new OWA(new LinealMaxWeighing(), new InstancesFromBag(x, X), getInstanceToClassLowerAppMembership(index, x, BB, CL)));
        break;
      case 2:
        if (allowSignature) name += "-Ave";
        M = new MembershipToClass(X, BB, CL,
            new Average(new InstancesFromBag(x, X), getInstanceToClassLowerAppMembership(index, x, BB, CL)));
        break;
      default:
        throw new IllegalArgumentException("No bag-to-class-lower-approximation membership match the supplied index");
    }
    return M;
  }

  /**
   * Creates a bag-to-class-upper-approximation membership function corresponding
   * to the <tt>index</tt>-th bag-to-class-upper-approximation membership
   * definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a bag-to-class-upper-approximation membership function.
   */
  private MembershipToClass getBagToClassUpperAppMembership(int index) {

    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> x = new Var();   // an instance
    Var <Instances> BB = new Var(); // the training samples

    MembershipToClass M = null;
    int bagToClassIndex = index / interSize2;

    switch (bagToClassIndex) {
      case 0:
        if (allowSignature) name += "-Max";
        M = new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X), getInstanceToClassUpperAppMembership(index, x, BB, CL)));
        break;
      case 1:
        if (allowSignature) name += "-OWAmax";
        M = new MembershipToClass(X, BB, CL,
            new OWA(new LinealMaxWeighing(), new InstancesFromBag(x, X), getInstanceToClassUpperAppMembership(index, x, BB, CL)));
        break;
      case 2:
        if (allowSignature) name += "-Ave";
        M = new MembershipToClass(X, BB, CL,
            new Average(new InstancesFromBag(x, X), getInstanceToClassUpperAppMembership(index, x, BB, CL)));
        break;
      default:
        throw new IllegalArgumentException("No bag-to-class-upper-approximation membership match the supplied index");
    }
    return M;
  }

  /**
   * Creates an instance-to-class-lower-approximation membership function
   * corresponding to the <tt>index</tt>-th instance-to-class-lower-approximation
   * membership definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @param x variable which is a holder for the instance which the membership
   * function refers to.
   * @param BB variable which is a holder for the set of training bags.
   * @param CL variable which is a holder for the class label which the membership
   * function refers to.
   * @return an instance-to-class-lower-approximation membership function of the
   * instance represented by x to the class represnted by CL in the training set
   * represented by BB.
   */
  private Evaluable getInstanceToClassLowerAppMembership(int index,
          Var <Instance> x, Var <Instances> BB, Var <Integer> CL) {

    Var <Instance> B = new Var();   // a bag
    Var <Instance> y = new Var();   // another instance

    int instanceToClassIndex = (index / interSize1) % numInstanceToClassAppMembership;

    switch (instanceToClassIndex) {
      case 0:
        if (allowSignature) name += "-STDminmax";
        return new Min(new BagsNotFromClass(B, BB, CL),
            new Min(new InstancesFromBag(y, B),
            new LukasiewiczImplicator(
            new CosineSimilarity(x, y), getInstanceToClassMembership(index, y, BB, CL))));
      case 1:
        if (allowSignature) name += "-OWAminmax";
        return new OWA(new LinealMinWeighing(), new BagsNotFromClass(B, BB, CL),
            new OWA(new LinealMinWeighing(), new InstancesFromBag(y, B),
            new LukasiewiczImplicator(
            new CosineSimilarity(x, y), getInstanceToClassMembership(index, y, BB, CL))));
      default:
        throw new IllegalArgumentException("No instance-to-class-lower-approximation membership match the supplied index");
    }
  }

  /**
   * Creates an instance-to-class-upper-approximation membership function
   * corresponding to the <tt>index</tt>-th instance-to-class-upper-approximation
   * membership definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @param x variable which is a holder for the instance which the membership
   * function refers to.
   * @param BB variable which is a holder for the set of training bags.
   * @param CL variable which is a holder for the class label which the membership
   * function refers to.
   * @return an instance-to-class-upper-approximation membership function of the
   * instance represented by x to the class represnted by CL in the training set
   * represented by BB.
   */
  private Evaluable getInstanceToClassUpperAppMembership(int index,
          Var <Instance> x, Var <Instances> BB, Var <Integer> CL) {

    Var <Instance> B = new Var();   // a bag
    Var <Instance> y = new Var();   // another instance

    int instanceToClassIndex = (index / interSize1) % numInstanceToClassAppMembership;

    switch (instanceToClassIndex) {
      case 0:
        if (allowSignature) name += "-STDminmax";
        return new Max(new BagsFromClass(B, BB, CL),
            new Max(new InstancesFromBag(y, B),
            new LukasiewiczTNorm(
            new CosineSimilarity(x, y), getInstanceToClassMembership(index, y, BB, CL))));
      case 1:
        if (allowSignature) name += "-OWAminmax";
        return new OWA(new LinealMaxWeighing(), new BagsFromClass(B, BB, CL),
            new OWA(new LinealMaxWeighing(), new InstancesFromBag(y, B),
            new LukasiewiczTNorm(
            new CosineSimilarity(x, y), getInstanceToClassMembership(index, y, BB, CL))));
      default:
        throw new IllegalArgumentException("No instance-to-class-upper-approximation membership match the supplied index");
    }
  }

  /**
   * Creates an instance-to-class membership function corresponding to the
   * <tt>index</tt>-th instance-to-class membership definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @param x variable which is a holder for the instance which the membership
   * function refers to.
   * @param BB variable which is a holder for the set of training bags.
   * @param CL variable which is a holder for the class label which the membership
   * function refers to.
   * @return an instance-to-class membership function of the instance represented
   * by x to the class represnted by CL in the training set represented by BB.
   */
  private Evaluable getInstanceToClassMembership(int index,
          Var <Instance> x, Var <Instances> BB, Var <Integer> CL) {

    Var <Instance> B = new Var();   // a bag

    int instanceToClassIndex = (index / numInstanceToBagMembership) % numInstanceToClassMembership;

    switch (instanceToClassIndex) {
      case 0:
        if (allowSignature) name += "-Ave";
        return new Average(new BagsFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B));
      case 1:
        if (allowSignature) name += "-OWAmax";
        return new OWA(new LinealMaxWeighing(), new BagsFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B));
      case 2:
        if (allowSignature) name += "-CompAve";
        return new Complement(
            new Average(new BagsNotFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B)));
      case 3:
        if (allowSignature) name += "-CompOWAmin";
        return new Complement(
            new OWA(new LinealMinWeighing(), new BagsNotFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B)));
      default:
        throw new IllegalArgumentException("No instance-to-class membership match the supplied index");
    }
  }

  /**
   * Creates an instance-to-bag membership function corresponding to the
   * <tt>index</tt>-th instance-to-bag membership definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @param x variable which is a holder for the instance which the membership
   * function refers to.
   * @param B variable which is a holder for the bag which the membership function
   * refers to.
   * @return an instance-to-bag membership function of the instance represented
   * by x to the bag represented by B.
   */
  private Evaluable getInstanceToBagMembership(int index,
          Var <Instance> x, Var <Instance> B) {

    Var <Instance> y = new Var();   // another instance

    int instanceToBagIndex = index % numInstanceToBagMembership;

    switch (instanceToBagIndex) {
      case 0:
        if (allowSignature) name += "-Max";
        return new Max(new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      case 1:
        if (allowSignature) name += "-OWAmax";
        return new OWA(new LinealMaxWeighing(), new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      case 2:
        if (allowSignature) name += "-Ave";
        return new Average(new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      default:
        throw new IllegalArgumentException("No instance-to-bag membership match the supplied index");
    }
  }

  /**
   * Size of the TFRIM family, based on the declared number of definitions.
   *
   * @return the size.
   */
  public int size() {
    return size;
  }

  // numBagToClassAppMembership * numInstanceToClassAppMembership * numInstanceToClassMembership * numInstanceToBagMembership
  // A numeric example:
  // numBagToClassAppMembership = 2
  // numInstanceToClassAppMembership = 2
  // numInstanceToClassMembership = 2
  // numInstanceToBagMembership = 3
  // interSize1 = 6
  // interSize2 = 12
  // size = 24
  // index =                    0 1 2 3 4 5 6 7 8 9 10  11  12  13  14  15  16  17  18  19  20  21  22  23
  // bagToClassAppIndex =       0 0 0 0 0 0 0 0 0 0 0   0   1   1   1   1   1   1   1   1   1   1   1   1
  // instanceToClassAppIndex =  0 0 0 0 0 0 1 1 1 1 1   1   0   0   0   0   0   0   1   1   1   1   1   1
  // instanceToClassIndex =     0 0 0 1 1 1 0 0 0 1 1   1   0   0   0   1   1   1   0   0   0   1   1   1
  // instanceToBagIndex =       0 1 2 0 1 2 0 1 2 0 1   2   0   1   2   0   1   2   0   1   2   0   1   2
  /**
   * Creates a TFRIM classifier corresponding to the <tt>index</tt>-th classifier
   * definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a Fuzzy-Rough Instance-based Multi-instance classifier.
   */
  public FRMClassifier buildClassifier(int index) {
    name = "TFRIM";
    allowSignature = false;
    MembershipToClass bagToClassLowerAppMembership = getBagToClassLowerAppMembership(index);
    allowSignature = true;
    MembershipToClass bagToClassUpperAppMembership = getBagToClassUpperAppMembership(index);
    FRMClassifier classifier = new FRMClassifier();
    classifier.setLowerAppMembership(bagToClassLowerAppMembership);
    classifier.setUpperAppMembership(bagToClassUpperAppMembership);
    classifier.setName(name);
    return classifier;
  }

}
