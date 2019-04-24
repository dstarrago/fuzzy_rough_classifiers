package oneClass.fuzzyRoughClassifiers;

import weka.core.Instance;
import weka.core.Instances;
import core.Evaluable;
import core.MembershipToClass;
import core.Var;
import core.iterators.BagsFromClass;
import core.iterators.BagsNotFromClass;
import core.metrics.AverageHausdorffSimilarity;
import core.metrics.HausdorffSimilarity;
import core.metrics.OWAAverageHausdorffSimilarity;
import core.metrics.OWAHausdorffSimilarity;
import core.multipleOperators.Max;
import core.multipleOperators.Min;
import core.multipleOperators.OWA;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;

/**
 * One-class Fuzzy-Rough Bag-based Multi-instance (OFRBM) classifier family factory.
 * Class for generating any classifier from the OFRBM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one OFRBM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the OFRBM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * OFRBMFactory f = new OFRBMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FRMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Sarah
 */
public class OFRBMFactory {
	
	/**
	   * Number of definitions of bag-to-class membership functions
	   */
	  public static final int numBagToClassMembership = 2;

	  /**
	   * Number of definitions of bag-based similarity functions
	   */
	  public static final int numBagBasedSimilarity = 4;

	  /**
	   * Size of the TFRBM family, based on the declared number of definitions.
	   */
	  private int size = numBagToClassMembership * numBagBasedSimilarity;

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
	    Var <Instances> BB = new Var(); // the training samples
	    Var <Instance> B = new Var();   // a bag

	    MembershipToClass M = null;
	    int bagToClassIndex = index / numBagBasedSimilarity;

	    switch (bagToClassIndex) {
	      case 0:
	        if (allowSignature) name += "-MaxCompMin";
	        M = new MembershipToClass(X, BB, CL,
	          new Complement(
	          new Min(new BagsNotFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B))));
	        break;
	      case 1:
	        if (allowSignature) name += "-OWAmaxCompOWAmin";
	        M = new MembershipToClass(X, BB, CL,
	          new Complement(
	          new OWA(new LinealMinWeighing(), new BagsNotFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B))));
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
	    Var <Instances> BB = new Var(); // the training samples
	    Var <Instance> B = new Var();   // a bag

	    MembershipToClass M = null;
	    int bagToClassIndex = index / numBagBasedSimilarity;

	    switch (bagToClassIndex) {
	      case 0:
	        if (allowSignature) name += "-MaxCompMin";
	        M = new MembershipToClass(X, BB, CL,
	          new Max(new BagsFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B)));
	        break;
	      case 1:
	        if (allowSignature) name += "-OWAmaxCompOWAmin";
	        M = new MembershipToClass(X, BB, CL,
	          new OWA(new LinealMaxWeighing(), new BagsFromClass(B, BB, CL), getBagBasedSimilarity(index, X, B)));
	        break;
	      default:
	        throw new IllegalArgumentException("No bag-to-class-upper-approximation membership match the supplied index");
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
	        if (allowSignature) name += "-Haus";
	        return new HausdorffSimilarity(X, B);
	      case 1:
	        if (allowSignature) name += "-OWAHaus";
	        return new OWAHausdorffSimilarity(X, B);
	      case 2:
	        if (allowSignature) name += "-AveHaus";
	        return new AverageHausdorffSimilarity(X, B);
	      case 3:
	        if (allowSignature) name += "-AveOWAHaus";
	        return new OWAAverageHausdorffSimilarity(X, B);
	      default:
	        throw new IllegalArgumentException("No bag-based similarity match the supplied index");
	    }
	  }

	  /**
	   * Size of the TFRBM family, based on the declared number of definitions.
	   *
	   * @return the size.
	   */
	  public int size() {
	    return size;
	  }

	  /**
	   * Creates a TFRBM classifier corresponding to the <tt>index</tt>-th classifier
	   * definition of the family.
	   *
	   * @param index is a number between 0 and <tt>size()-1</tt>.
	   * @return a Fuzzy-Rough Bag-based Multi-instance classifier.
	   */
	  public OFRMClassifier buildClassifier(int index) {
	    name = "OFRBM";
	    allowSignature = false;
	    MembershipToClass bagToClassLowerAppMembership = getBagToClassLowerAppMembership(index);
	    allowSignature = true;
	    MembershipToClass bagToClassUpperAppMembership = getBagToClassUpperAppMembership(index);
	    EmptyOFRMClassifier classifier = new EmptyOFRMClassifier();
	    classifier.setLowerAppMembership(bagToClassLowerAppMembership);
	    classifier.setUpperAppMembership(bagToClassUpperAppMembership);
	    classifier.setName(name);
	    return classifier;
	  }

}
