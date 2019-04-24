package twoClasses.fuzzyClassifiers;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import core.MembershipToClass;
import core.Var;
import core.iterators.BagsFromClass;
import core.metrics.AverageHausdorffSimilarity;
import core.metrics.HausdorffSimilarity;
import core.metrics.NormChamferWindowSimilarity;
import core.multipleOperators.Average;
import core.multipleOperators.OWA;
import core.owaWeighing.ExponentialMaxWeighing;
import core.owaWeighing.InverseAddMaxWeighing;
import core.owaWeighing.LinealMaxWeighing;


/**
 * Fuzzy bag based multi-instance classifiers using:
 * - membership function of bag to class: OWAMax, with exponential weights
 * - Similarity between bags: AveHaus
 * - Similarity between instances: cosine
 *
 * @author Sarah
 */
public class FBExp extends FMClassifier implements Serializable {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  /**
	   * Creates a FBM1 classifier by simply supplying its definition.
	   */
	  public FBExp() {
	    Var <Integer> CL = new Var();   // target class label
	    Var <Instance> X = new Var();   // bag with unknown label
	    Var <Instance> B = new Var();   // a bag
	    Var <Instances> BB = new Var(); // the training samples
	    
	    
	    //MembershipToClass M = new MembershipToClass(X, BB, CL,
	      //      new OWA(new ExponentialMaxWeighing(), new BagsFromClass(B, BB, CL), new AverageHausdorffSimilarity(X, B)));
	    
	    MembershipToClass M = new MembershipToClass(X, BB, CL,
	  	            new OWA(new InverseAddMaxWeighing(), new BagsFromClass(B, BB, CL), new NormChamferWindowSimilarity(X, B)));

	    setMembership(M);
	  }

}
