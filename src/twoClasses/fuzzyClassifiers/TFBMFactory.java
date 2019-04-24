/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package twoClasses.fuzzyClassifiers;

import core.multipleOperators.*;
import core.iterators.*;
import core.Evaluable;
import core.Var;
import core.metrics.*;
import core.MembershipToClass;
import core.owaWeighing.InverseAddMaxWeighing;
import core.owaWeighing.InverseAddMinWeighing;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Two-class Fuzzy Bag-based Multi-instance (TFBM) classifier family factory.
 * Class for generating any classifier from the TFBM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one TFBM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the TFBM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * TFBMFactory f = new TFBMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Danel
 */
public class TFBMFactory {

  /**
   * Number of definitions of bag-to-class membership functions
   */
  //private final int numBagToClassMembership = 4;
  private final int numBagToClassMembership = 6;

  /**
   * Number of definitions of bag-based similarity functions
   */
  //private final int numBagBasedSimilarity = 4;
  private final int numBagBasedSimilarity = 6;

  /**
   * Size of the TFBM family, based on the declared number of definitions.
   */
  private final int size = numBagToClassMembership * numBagBasedSimilarity;

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
    Var <Instances> BB = new Var(); // the training samples
    Var <Instance> B = new Var();   // a bag

    MembershipToClass M = null;
    int bagToClassIndex = index / numBagBasedSimilarity;

    switch (bagToClassIndex) {
      case 0:
        name += "-Ave";
        M = new MembershipToClass(X, BB, CL,
          new Average(new BagsFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B)));
        break;
      case 1:
        name += "-OWAmax";
        M = new MembershipToClass(X, BB, CL,
          new OWA(new LinealMaxWeighing(), new BagsFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B)));
        break;
      case 2:
        name += "-CompAve";
        M = new MembershipToClass(X, BB, CL,
          new Complement(
            new Average(new BagsNotFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B))));
        break;
      case 3:
        name += "-CompOWAmin";
        M = new MembershipToClass(X, BB, CL,
          new Complement(
            new OWA(new LinealMinWeighing(), new BagsNotFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B))));
        break;
      case 4:
          name += "-OWAmaxAdd";
          M = new MembershipToClass(X, BB, CL,
            new OWA(new InverseAddMaxWeighing(), new BagsFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B)));
          break;       
      case 5:
          name += "-CompOWAminAdd";
          M = new MembershipToClass(X, BB, CL,
            new Complement(
              new OWA(new InverseAddMinWeighing(), new BagsNotFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B))));
          break; 
        
        
        
      default:
        throw new IllegalArgumentException("No bag-to-class membership match the supplied index");
    }
    return M;
  }

  /**
   * Creates a bag-based similarity function corresponding to the
   * <tt>index</tt>-th bag-based similarity definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @param X variable which is a holder for the bag subject of the similarity.
   * @param B variable which is a holder for the comparative bag.
   * @return a bag-based similarity function between the bags referenced by the
   * variables X and B.
   */
  private Evaluable getBagBasedSimilarity(int index,
          Var <Instance> X, Var <Instance> B) {

    int instanceToBagIndex = index % numBagBasedSimilarity;

    switch (instanceToBagIndex) {
      case 0:
        name += "-Haus";
        return new HausdorffSimilarity(X, B);
      case 1:
        name += "-OWAHaus";
        return new OWAHausdorffSimilarity(X, B);
      case 2:
        name += "-AveHaus";
        return new AverageHausdorffSimilarity(X, B);
      case 3:
        name += "-AveOWAHaus";
        return new OWAAverageHausdorffSimilarity(X, B);
//      case 4:
//          name += "-NormChamf";
//          return new NormChamferSimilarity(X, B);
//      case 5:
//          name += "-OWANormChamf";
//          return new OWANormChamferSimilarity(X, B);          
      case 4:
          name += "-OWAHausAdd";
          return new OWAHausdorffSimilarity(new InverseAddMaxWeighing(), new InverseAddMinWeighing(), 
        		  CosineDistance.class, X, B);    
      case 5:
          name += "-AveOWAHausAdd";
          return new OWAAverageHausdorffSimilarity(new InverseAddMinWeighing(), CosineDistance.class,X, B);       
//      case 8:
//          name += "-OWANormChamfAdd";
//          return new OWANormChamferSimilarity(new InverseAddMinWeighing(), CosineDistance.class, X, B); 
               
   
      default:
        throw new IllegalArgumentException("No bag-based similarity match the supplied index");
    }
  }

  /**
   * Size of the TFBM family, based on the declared number of definitions.
   *
   * @return the size.
   */
  public int size() {
    return size;
  }

  // A numeric example:
  // instanceToClassMembershipList.size() = 2
  // instanceToBagMembershipList.size() = 3
  // bagToClassMembershipList.size() = 2
  // interSize = 6
  // size = 12
  // index =                0 1 2 3 4 5 6 7 8 9 10  11
  // bagToClassIndex =      0 0 0 0 0 0 1 1 1 1 1   1
  // instanceToClassIndex = 0 0 0 1 1 1 0 0 0 1 1   1
  // instanceToBagIndex =   0 1 2 0 1 2 0 1 2 0 1   2
  /**
   * Creates a TFBM classifier corresponding to the <tt>index</tt>-th classifier
   * definition of the family.
   *
   * @param index is a number between 0 and <tt>size()-1</tt>.
   * @return a Fuzzy Bag-based Multi-instance classifier.
   */
  public FMClassifier buildClassifier(int index) {
    name = "TFBM";
    MembershipToClass bagToClassMembership = getBagToClassMembership(index);
    FMClassifier classifier = new FMClassifier();
    classifier.setMembership(bagToClassMembership);
    classifier.setName(name);
    return classifier;
  }

}
