package oneClass.fuzzyClassifiers;

import weka.core.Instance;
import weka.core.Instances;
import core.Evaluable;
import core.MembershipToClass;
import core.Var;
import core.iterators.BagsFromClass;
import core.iterators.BagsNotFromClass;
import core.iterators.InstancesFromBag;
import core.metrics.CosineSimilarity;
import core.multipleOperators.Average;
import core.multipleOperators.Max;
import core.multipleOperators.OWA;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.unaryOperators.Complement;


/**
 * One-class Fuzzy Instance-based Multi-instance (OFIM) classifier family factory.
 * Class for generating any classifier from the OFIM family.
 * <p>
 *
 * Internally contains several membership function definitions, each representing
 * one OFIM classifier. The number of classifier definitions in the family is given
 * by <tt>size()</tt>. Use <tt>buildClassifier(int index)</tt> to create the
 * <tt>index</tt>-th classifier type in the list of the TFIM family.
 * <p>
 *
 * Example usage:
 * <p>
 * <code>
 * OFIMFactory f = new OFIMFactory(); <br>
 * for (int i = 0; i < f.size(); i++) { <br>
 *   FMClassifier c = f.buildClassifier(i); <br>
 *   System.out.println(c.name()); <br>
 * } <br>
 * </code>
 *
 * @author Sarah
 */
public class OFIMFactory {
	
	/**
	   * Number of definitions of bag-to-class membership functions
	   */
	  private final int numBagToClassMembership = 3;

	  /**
	   * Number of definitions of instance-to-class membership functions
	   */
	  private final int numInstanceToClassMembership = 4;

	  /**
	   * Number of definitions of instance-to-bag membership functions
	   */
	  private final int numInstanceToBagMembership = 3;

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
	   * by x to the class represented by CL in the training set represented by BB.
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

	  /**
	   * Creates a OFIM classifier corresponding to the <tt>index</tt>-th classifier
	   * definition of the family.
	   *
	   * @param index is a number between 0 and <tt>size()-1</tt>.
	   * @return a Fuzzy Instance-based Multi-instance classifier.
	   */
	  public OFMClassifier buildClassifier(int index) {
	    name = "OFIM";
	    MembershipToClass bagToClassMembership = getBagToClassMembership(index);
	    EmptyOFMClassifier classifier = new EmptyOFMClassifier();
	    classifier.setMembership(bagToClassMembership);
	    classifier.setName(name);
	    return classifier;
	  }

}
