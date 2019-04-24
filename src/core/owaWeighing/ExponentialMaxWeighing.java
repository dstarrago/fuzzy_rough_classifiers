package core.owaWeighing;

import java.io.Serializable;

import core.MaxWeighingMaker;

/**
 * Class for generating a maximum OWA-weight vector, using exponential weights.
 *
 * @author Sarah
 */
public class ExponentialMaxWeighing extends MaxWeighingMaker implements Serializable {
	
	/** for serialization */
	private static final long serialVersionUID = 1L;

	public double[] getWeightVector(int size) {
	    double[] weights = new double[size];
	    double p = size;
	    		
	    double denom = Math.pow(2.0,  p) - 1.0;
	    
	    // determine weights
	    for (int i = 1; i <= size; i++) {
	      weights[i - 1] = Math.pow(2.0, p - i) / denom;
	    }
	    
	    return weights;
	}

}
