/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import core.multipleOperators.*;
import core.iterators.*;
import core.Evaluable;
import core.Var;
import core.metrics.CosineSimilarity;
import core.MembershipToClass;
import core.owaWeighing.InverseAddMaxWeighing;
import core.owaWeighing.InverseAddMinWeighing;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Two-class Fuzzy Instance-based Multi-instance (TFIM) classifier family factory.
 * Class for generating any classifier from the TFIM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one TFIM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the TFIM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * TFIMFactory f = new TFIMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Danel
 */
public class TFIMFactory {

  /**
   * Number of definitions of bag-to-class membership functions
   */
  private final int numBagToClassMembership = 4;

  /**
   * Number of definitions of instance-to-class membership functions
   */
  private final int numInstanceToClassMembership = 6;

  /**
   * Number of definitions of instance-to-bag membership functions
   */
  private final int numInstanceToBagMembership = 4;

  /**
   * Number of combinations of the last two terms in the family membership
   * function definition.
   */
  private final int interSize = numInstanceToBagMembership * numInstanceToClassMembership;

  /**
   * Size of the TFIM family, based on the declared number of definitions.
   */
  private final int size = numBagToClassMembership * interSize;

  /**
   * For composing the name of the generated classifier.
   */
  private String name;

  /**
   * Creates a bag-to-class membership function corresponding to the
   * <tt>index</tt>-th bag-to-class membership definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a bag-to-class membership function.
   */
  private MembershipToClass getBagToClassMembership(int index) {

    Var <Integer> CL = new Var();   // target class label
    Var <Instance> X = new Var();   // bag with unknown label
    Var <Instance> x = new Var();   // an instance
    Var <Instances> BB = new Var(); // the training samples

    MembershipToClass M = null;
    int bagToClassIndex = index / interSize;

    switch (bagToClassIndex) {
      case 0:
        name += "-Max";
        M = new MembershipToClass(X, BB, CL,
            new Max(new InstancesFromBag(x, X), getInstanceToClassMembership(index, x, BB, CL)));
        break;
      case 1:
        name += "-OWAmax";
        M = new MembershipToClass(X, BB, CL,
            new OWA(new LinealMaxWeighing(), new InstancesFromBag(x, X), getInstanceToClassMembership(index, x, BB, CL)));
        break;
      case 2:
        name += "-Ave";
        M = new MembershipToClass(X, BB, CL,
            new Average(new InstancesFromBag(x, X), getInstanceToClassMembership(index, x, BB, CL)));
        break;
      case 3:
          name += "-OWAmaxAdd";
          M = new MembershipToClass(X, BB, CL,
              new OWA(new InverseAddMaxWeighing(), new InstancesFromBag(x, X), getInstanceToClassMembership(index, x, BB, CL)));
          break;  
        
        
      default:
        throw new IllegalArgumentException("No bag-to-class membership match the supplied index");
    }
    return M;
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
        name += "-Ave";
        return new Average(new BagsFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B));
      case 1:
        name += "-OWAmax";
        return new OWA(new LinealMaxWeighing(), new BagsFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B));
      case 2:
        name += "-CompAve";
        return new Complement(
            new Average(new BagsNotFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B)));
      case 3:
        name += "-CompOWAmin";
        return new Complement(
            new OWA(new LinealMinWeighing(), new BagsNotFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B)));        
      case 4:
          name += "-OWAmaxAdd";
          return new OWA(new InverseAddMaxWeighing(), new BagsFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B)); 
      case 5:
          name += "-CompOWAminAdd";
          return new Complement(
              new OWA(new InverseAddMinWeighing(), new BagsNotFromClass(B, BB, CL), getInstanceToBagMembership(index, x, B))); 
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
        name += "-Max";
        return new Max(new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      case 1:
        name += "-OWAmax";
        return new OWA(new LinealMaxWeighing(), new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      case 2:
        name += "-Ave";
        return new Average(new InstancesFromBag(y, B), new CosineSimilarity(x, y));
      case 3:
          name += "-OWAmaxAdd";
          return new OWA(new InverseAddMaxWeighing(), new InstancesFromBag(y, B), new CosineSimilarity(x, y));        
        
      default:
        throw new IllegalArgumentException("No instance-to-bag membership match the supplied index");
    }
  }

  /**
   * Size of the TFIM family, based on the declared number of definitions.
   *
   * @return the size.
   */
  public int size() {
    return size;
  }

  // numBagToClassMembership * numInstanceToClassMembership * numInstanceToBagMembership
  // A numeric example:
  // bagToClassMembershipList.size() = 2
  // instanceToClassMembershipList.size() = 2
  // instanceToBagMembershipList.size() = 3
  // interSize = 6
  // size = 12
  // index =                0 1 2 3 4 5 6 7 8 9 10  11
  // bagToClassIndex =      0 0 0 0 0 0 1 1 1 1 1   1
  // instanceToClassIndex = 0 0 0 1 1 1 0 0 0 1 1   1
  // instanceToBagIndex =   0 1 2 0 1 2 0 1 2 0 1   2
  /**
   * Creates a TFIM classifier corresponding to the <tt>index</tt>-th classifier
   * definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a Fuzzy Instance-based Multi-instance classifier.
   */
  public FMClassifier buildClassifier(int index) {
    name = "TFIM";
    MembershipToClass bagToClassMembership = getBagToClassMembership(index);
    FMClassifier classifier = new FMClassifier();
    classifier.setMembership(bagToClassMembership);
    classifier.setName(name);
    return classifier;
  }

}
