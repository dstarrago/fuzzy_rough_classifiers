package core.owaWeighing;

import java.io.Serializable;

import core.MaxWeighingMaker;

/**
 * Class for generating a windowing OWA-weight vector.
 * It averages over the middle 60% of the supplied instances.
 * 
 *     = 0    , if i < k
 * w_i = 1/m  , if k <= i < k+m
 *     = 0    , if i >= k + m
 *
 * @author Sarah
 */
public class WindowAverage extends MaxWeighingMaker implements Serializable {
	
	/** for serialization */
	private static final long serialVersionUID = 1L;

	public double[] getWeightVector(int size) {
	    double[] weights = new double[size];
	    double p = size;
	    		
	    int k = (int) Math.round(p / 5.0);
	    int m = (int) Math.round(p * 3.0 / 5.0);
	    
	    // determine weights
	    for (int i = 0; i < k; i++) {
	      weights[i] = 0;
	    }
	    
	    for (int i = k; i < k+m; i++) {
		      weights[i] = (double) 1.0 / m;
	    }
	    
	    for (int i = k+m; i < size; i++) {
		      weights[i] = 0;
	    }
	    
	    return weights;
	}

}
