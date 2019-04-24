package oneClass.fuzzyClassifiers;

import weka.core.Instance;
import weka.core.Instances;
import core.Evaluable;
import core.MembershipToClass;
import core.Var;
import core.iterators.BagsFromClass;
import core.iterators.BagsNotFromClass;
import core.metrics.AverageHausdorffSimilarity;
import core.metrics.CosineDistance;
import core.metrics.HausdorffSimilarity;
import core.metrics.NormChamferSimilarity;
import core.metrics.OWAAverageHausdorffSimilarity;
import core.metrics.OWAHausdorffSimilarity;
import core.metrics.OWANormChamferSimilarity;
import core.multipleOperators.Average;
import core.multipleOperators.OWA;
import core.owaWeighing.InverseAddMaxWeighing;
import core.owaWeighing.InverseAddMinWeighing;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;


/**
 * One-class Fuzzy Bag-based Multi-instance (OFBM) classifier family factory.
 * Class for generating any classifier from the OFBM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one OFBM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the OFBM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * OFBMFactory f = new OFBMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Sarah
 */
public class OFBMFactory {
	
	/**
	   * Number of definitions of bag-to-class membership functions
	   */
	  private final int numBagToClassMembership = 6;

	  /**
	   * Number of definitions of bag-based similarity functions
	   */
	  private final int numBagBasedSimilarity = 9;

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
	      case 4:
	          name += "-NormChamf";
	          return new NormChamferSimilarity(X, B);
	      case 5:
	          name += "-OWANormChamf";
	          return new OWANormChamferSimilarity(X, B);          
	      case 6:
	          name += "-OWAHausAdd";
	          return new OWAHausdorffSimilarity(new InverseAddMaxWeighing(), new InverseAddMinWeighing(), 
	        		  CosineDistance.class, X, B);    
	      case 7:
	          name += "-AveOWAHausAdd";
	          return new OWAAverageHausdorffSimilarity(new InverseAddMinWeighing(), CosineDistance.class, X, B);       
	      case 8:
	          name += "-OWANormChamfAdd";
	          return new OWANormChamferSimilarity(new InverseAddMinWeighing(), CosineDistance.class, X, B);    
	   
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

	  /**
	   * Creates a OFBM classifier corresponding to the <tt>index</tt>-th classifier
	   * definition of the family.
	   *
	   * @param index is a number between 0 and <tt>size()-1</tt>.
	   * @return a Fuzzy Bag-based Multi-instance classifier.
	   */
	  public OFMClassifier buildClassifier(int index) {
	    name = "OFBM";
	    MembershipToClass bagToClassMembership = getBagToClassMembership(index);
	    EmptyOFMClassifier classifier = new EmptyOFMClassifier();
	    classifier.setMembership(bagToClassMembership);
	    classifier.setName(name);
	    return classifier;
	  }

}
