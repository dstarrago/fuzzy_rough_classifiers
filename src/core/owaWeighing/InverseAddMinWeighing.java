package core.owaWeighing;

import java.io.Serializable;

import core.MinWeighingMaker;

/**
 * Class for generating a minimum OWA-weight vector, using inverse additive weights.
 *
 * @author Sarah
 */
public class InverseAddMinWeighing extends MinWeighingMaker implements Serializable {
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  public double[] getWeightVector(int size) {
	    double[] weights = new double[size];
	    double p = size;
	    
	    // sum for denominator
	    double sum = 0.0;
	    for(int i = 1; i <= p; i++){
	    	sum += (double) 1/i;
	    }
	    
	    // determine weights
	    for (int i = 1; i <= size; i++) {
	      weights[i - 1] = 1 / ((p - i + 1) * sum);
	    }
	    
	    return weights;
	  }

}
