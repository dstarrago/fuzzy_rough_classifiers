package core.owaWeighing;

import java.io.Serializable;

import core.MinWeighingMaker;

/**
 * Class for generating a minimum OWA-weight vector, using inverse additive weights.
 *
 * @author Sarah
 */
public class ExponentialMinWeighing extends MinWeighingMaker implements Serializable {

	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  public double[] getWeightVector(int size) {
	    double[] weights = new double[size];
	    double p = size;
	    
	    double denom = Math.pow(2.0,  p) - 1.0;
	    
	    // determine weights
	    for (int i = 1; i <= size; i++) {
	      weights[i - 1] = Math.pow(2.0, i - 1) / denom;
	    }
	    
	    return weights;
	  }	
}
