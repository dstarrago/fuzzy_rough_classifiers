package twoClasses.fuzzyClassifiers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities.Capability;

public class TFIMFactory_Sarah extends AbstractClassifier implements Serializable{
	
	
	private String instanceToBagMembership;      // "-Max", "-MaxL", "-MaxIA", "-Avg"
	private String instanceToClassMembership;    // "-Avg", "-OWAL", "-OWAIA", "-CAvg", "-CAvgL", "-CAvgIA"
	private String bagToClassMembership;      // "-Max", "-MaxL", "-MaxIA", "-Avg"
	
	
	private double[] attrMin;
	private double[] attrMax;
	


	  /**
	   * The training data
	   */
	  private Instances train;

	  /** for serialization */
	  private static final long serialVersionUID = 1L;
	  
	  
	  public TFIMFactory_Sarah(String instanceToBagMembership, String instanceToClassMembership,
			  String bagToClassMembership){
		  this.bagToClassMembership = bagToClassMembership;
		  this.instanceToClassMembership = instanceToClassMembership;
		  this.instanceToBagMembership = instanceToBagMembership;
	  }
	  

	  /**
	   * Builds the classifier
	   *
	   * @param train the training data
	   * @throws Exception if the classifier could not be built successfully
	   */
	  public void buildClassifier(Instances data) throws Exception {

	    // can classifier handle the data?
	    getCapabilities().testWithFail(data);

	    // remove instances with missing class
	    train = new Instances(data);
	    train.deleteWithMissingClass();
	    
	    // precompute attribute ranges
	    precompute();
	    
	  }
	  


	  /**
	   * Computes the Euclidean similarity between two instances.
	   *
	   * @param a the first instance in the comparison.
	   * @param b the second instance in the comparison.
	   * @return number between 0 and 1 representing the cosine similarity.
	   */
	  private double instanceSimilarity(Instance a, Instance b) {
	    double sim = 0.0;
	    for (int i = 0; i < a.numAttributes(); i++) {	    	
	    	if(a.attribute(i).isNumeric()){
	    		
	    		double aVal = a.value(i);
	    		double bVal = b.value(i);
	    		
	    		// modify range if necessary
	    		if(aVal < attrMin[i]){
	    			attrMin[i] = aVal;
	    		} 
	    		if(aVal > attrMax[i]){
	    			attrMax[i] = aVal;
	    		}
	    		if(bVal < attrMin[i]){
	    			attrMin[i] = bVal;
	    		} 
	    		if(bVal > attrMax[i]){
	    			attrMax[i] = bVal;
	    		}
	    			    		
	    		// normalize
    			double range = attrMax[i] - attrMin[i];
	    		
	    		if(range != 0){
		    		double aNorm = (aVal - attrMin[i]) / range;
		    		double bNorm = (bVal - attrMin[i]) / range;
		    		
		    		double diff = aNorm - bNorm;
	    		
		    		// add to distance
		    		sim += 1.0 - diff * diff;
	    		} else {
	    			sim++;
	    		}
	    		
	    	} else {
	    		if(a.value(i) == b.value(i)){
	    			sim++;
	    		}
	    	}
	    }
    
	    
	    return sim / attrMax.length;
	  }

		  /**
		   * Calculates the membership degree of instance x to the class with label
		   * <tt>classLabel</tt>.
		   *
		   * @param x instance which membership degree has to be calculated.
		   * @param classLabel index of the class.
		   * @return the class membership.
		   */
		  private double instanceToClassMembership(Instance x, int classLabel) {
			  
		  
			  
		      ////////////////////////////////////////////
		      //////// Average to same-class bags ////////	
	          ////////////////////////////////////////////
			  if(instanceToClassMembership.equals("-Avg")){			  
				  
			    double avg = 0;
			    int count = 0;
			    for (int j = 0; j < train.numInstances(); j++) {
			      Instance B = train.instance(j);
			      if (B.classValue() != classLabel) continue;		      
			      
			      // determine membership of instance to bag
			      double memToBag;
			      if(instanceToBagMembership.equals("-Max")){
			    	  
			    	  memToBag = 0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        double R = instanceSimilarity(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-MaxL")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Avg")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += instanceSimilarity(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-MaxIA")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("maxadd", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else {
			    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
			      }
			      
			      // add to average and increase count
			      avg += memToBag;
			      count++;
			    }
			    avg /= count;
			    return avg;

			  } 
		      ////////////////////////////////////////////
		      //////// OWAmax to same-class bags /////////	
	          ////////////////////////////////////////////
			  else if(instanceToClassMembership.equals("-OWAL")){			  
				  
				ArrayList<Double> values = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
			      Instance B = train.instance(j);
			      if (B.classValue() != classLabel) continue;		      
			      
			      // determine membership of instance to bag
			      double memToBag;
			      if(instanceToBagMembership.equals("-Max")){
			    	  
			    	  memToBag = 0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        double R = instanceSimilarity(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-MaxL")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Avg")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += instanceSimilarity(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-MaxIA")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("maxadd", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else {
			    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
			      }
			      
			      // add to list
			      values.add(memToBag);
			    }
			    Collections.sort(values, Collections.reverseOrder()); // decreasing
			    double[] weights = getWeights("max", values.size());
			    
			    double sum = 0.0;
			    for(int i = 0; i < values.size(); i++){
			    	  sum += weights[i] * values.get(i);
			      }
			    return sum;
				  
			  } 
		      ///////////////////////////////////////////////////////////
		      //////// Complement of average to other-class bags ////////	
	          ///////////////////////////////////////////////////////////		  
			  else if(instanceToClassMembership.equals("-CAvg")){ 
	  
				double avg = 0;
			    int count = 0;
			    for (int j = 0; j < train.numInstances(); j++) {
			      Instance B = train.instance(j);
			      if (B.classValue() == classLabel) continue;	// opposite class only	      
			      
			      // determine membership of instance to bag
			      double memToBag;
			      if(instanceToBagMembership.equals("-Max")){
			    	  
			    	  memToBag = 0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        double R = instanceSimilarity(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-MaxL")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Avg")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += instanceSimilarity(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-MaxIA")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("maxadd", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else {
			    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
			      }
			      
			      // add to average and increase count
			      avg += memToBag;
			      count++;
			    }
			    avg /= count;
			    return 1.0 - avg;  
	  
				  
			  } 
		      //////////////////////////////////////////////////////////
		      //////// Complement of OWAmin to other-class bags ////////	
	          //////////////////////////////////////////////////////////		  
			  else if(instanceToClassMembership.equals("-CAvgL")){
				  
				ArrayList<Double> values = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
			      Instance B = train.instance(j);
			      if (B.classValue() == classLabel) continue;	// only other-class instances	      
			      
			      // determine membership of instance to bag
			      double memToBag;
			      if(instanceToBagMembership.equals("-Max")){
			    	  
			    	  memToBag = 0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        double R = instanceSimilarity(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-MaxL")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Avg")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += instanceSimilarity(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-MaxIA")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("maxadd", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else {
			    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
			      }
			      
			      // add to list
			      values.add(memToBag);
			    }
			    Collections.sort(values, Collections.reverseOrder()); // decreasing
			    double[] weights = getWeights("min", values.size());
			    
			    double sum = 0.0;
			    for(int i = 0; i < values.size(); i++){
			    	  sum += weights[i] * values.get(i);
			      }
			    return 1.0 - sum;
				  
			  } 
		      ///////////////////////////////////////////////
		      //////// OWAmaxAdd to same-class bags /////////	
	          ///////////////////////////////////////////////		  
			  else if(instanceToClassMembership.equals("-OWAIA")){
				  
				ArrayList<Double> values = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
			      Instance B = train.instance(j);
			      if (B.classValue() != classLabel) continue;		      
			      
			      // determine membership of instance to bag
			      double memToBag;
			      if(instanceToBagMembership.equals("-Max")){
			    	  
			    	  memToBag = 0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        double R = instanceSimilarity(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-MaxL")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Avg")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += instanceSimilarity(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-MaxIA")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(instanceSimilarity(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("maxadd", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else {
			    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
			      }
			      
			      // add to list
			      values.add(memToBag);
			    }
			    Collections.sort(values, Collections.reverseOrder()); // decreasing
			    double[] weights = getWeights("maxadd", values.size());
			    
			    double sum = 0.0;
			    for(int i = 0; i < values.size(); i++){
			    	  sum += weights[i] * values.get(i);
			      }
			    return sum;			  
				  
			  } 
		      ///////////////////////////////////////////////////////////
		      //////// Complement of OWAminAdd to other-class bags //////	
	          ///////////////////////////////////////////////////////////
			  else if(instanceToClassMembership.equals("-CAvgIA")){
				  
				  ArrayList<Double> values = new ArrayList<Double>();
				    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;	// only other-class instances	      
				      
				      // determine membership of instance to bag
				      double memToBag;
				      if(instanceToBagMembership.equals("-Max")){
				    	  
				    	  memToBag = 0;
					      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
					        Instance y = B.relationalValue(1).instance(k);
					        double R = instanceSimilarity(x, y);
					        if (R > memToBag)
					        	memToBag = R;
					      }
					      
				      } else if(instanceToBagMembership.equals("-MaxL")){
				    	  
				    	  ArrayList<Double> allValues = new ArrayList<Double>();
				    	  
					      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
					        Instance y = B.relationalValue(1).instance(k);
					        allValues.add(instanceSimilarity(x, y));
					      }
					      
					      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
					      double[] weights = getWeights("max", allValues.size());
					      memToBag = 0.0;
					      for(int i = 0; i < allValues.size(); i++){
					    	  memToBag += weights[i] * allValues.get(i);
					      }
				    	  
				      } else if(instanceToBagMembership.equals("-Avg")){
				    	  
				    	  memToBag = 0.0;
					      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
					        Instance y = B.relationalValue(1).instance(k);
					        memToBag += instanceSimilarity(x, y);
					      }
					      memToBag /= B.relationalValue(1).numInstances();
				    	  
				      } else if(instanceToBagMembership.equals("-MaxIA")){
				    	  
				    	  ArrayList<Double> allValues = new ArrayList<Double>();
				    	  
					      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
					        Instance y = B.relationalValue(1).instance(k);
					        allValues.add(instanceSimilarity(x, y));
					      }
					      
					      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
					      double[] weights = getWeights("maxadd", allValues.size());
					      memToBag = 0.0;
					      for(int i = 0; i < allValues.size(); i++){
					    	  memToBag += weights[i] * allValues.get(i);
					      }
				    	  
				      } else {
				    	  throw new IllegalArgumentException("Invalid instance-to-bag membership."); 
				      }
				      
				      // add to list
				      values.add(memToBag);
				    }
				    Collections.sort(values, Collections.reverseOrder()); // decreasing
				    double[] weights = getWeights("minadd", values.size());
				    
				    double sum = 0.0;
				    for(int i = 0; i < values.size(); i++){
				    	  sum += weights[i] * values.get(i);
				      }
				    return 1.0 - sum;
				  
			  } 
		      /////////////////////////
		      //////// Invalid ////////	
	          /////////////////////////		  
			  else {
				  throw new IllegalArgumentException("Invalid instance-to-class membership.");
			  }
		  }



		  /**
		   * Computes the membership degree of a bag X to the given class.
		   *
		   * @param X the bag.
		   * @param classLabel label of the given class.
		   * @return the membership degree of a bag X to the lower approximation of
		   * the given class.
		   */
		  private double membership(Instance X, int classLabel) {
			  
			  
		      /////////////////////////
			  //////// Maximum ////////
	          /////////////////////////
			  if(bagToClassMembership.equals("-Max")){
		    
				    double max = 0;
				    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
				      Instance x = X.relationalValue(1).instance(i);
				      
				      double thisValue = instanceToClassMembership(x, classLabel);
				      
				      if (thisValue > max)
				        max = thisValue;
				    }
				    return max;
				    			    
			  } 
		      /////////////////////////////
			  //////// OWA-Maximum ////////
	          /////////////////////////////		  
			  else if(bagToClassMembership.equals("-MaxL")){			  
				  
				    ArrayList<Double> values = new ArrayList<Double>();
				    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
				      Instance x = X.relationalValue(1).instance(i);				      
				      values.add(instanceToClassMembership(x, classLabel));
				    }
				    
				    Collections.sort(values, Collections.reverseOrder());
				    double[] weights = getWeights("max", values.size());
				    double res = 0.0;
				    for(int i = 0; i < values.size(); i++){
				    	res += weights[i] * values.get(i);
				    }
				    return res;

			  } 
		      /////////////////////////
			  //////// Average ////////
	          /////////////////////////		  
			  else if(bagToClassMembership.equals("-Avg")){

				    double avg = 0;
				    int count = 0;
				    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
				      Instance x = X.relationalValue(1).instance(i);
				      avg += instanceToClassMembership(x, classLabel);
				      count++;
				    }
				    return (double) avg / count;			  

			  } 
		      ////////////////////////////
			  //////// OWA-MaxAdd ////////
	          ////////////////////////////	
			  else if(bagToClassMembership.equals("-MaxIA")){
				  
				    ArrayList<Double> values = new ArrayList<Double>();
				    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
				      Instance x = X.relationalValue(1).instance(i);
				      values.add(instanceToClassMembership(x, classLabel));
				    }
				    
				    Collections.sort(values, Collections.reverseOrder());
				    double[] weights = getWeights("maxadd", values.size());
				    double res = 0.0;
				    for(int i = 0; i < values.size(); i++){
				    	res += weights[i] * values.get(i);
				    }
				    return res;	
				  
			  } else {
				  throw new IllegalArgumentException("Invalid bag-to-class approximation membership.");
			  }
		  }

		  

		  /**
		   * Computes the distribution for a given exemplar
		   *
		   * @param exmp the exemplar for which distribution is computed
		   * @return the distribution
		   * @throws Exception if the distribution can't be computed successfully
		   */
		  @Override
		  public double[] distributionForInstance(Instance exmp) throws Exception {
		    int numClasses = exmp.dataset().classAttribute().numValues();
		    double [] distribution = new double[numClasses];

		    for (int i = 0; i < numClasses; i++) {
		      // Calculates the final distribution
		      distribution[i] = membership(exmp, i);
		    }
		    return distribution;
		  }

		  /**
		   * Returns default capabilities of the classifier.
		   *
		   * @return the capabilities of this classifier
		   */
		  @Override
		  public Capabilities getCapabilities() {
		    Capabilities result = super.getCapabilities();
		    result.disableAll();

		    // attributes
		    result.enable(Capability.NOMINAL_ATTRIBUTES);
		    result.enable(Capability.RELATIONAL_ATTRIBUTES);
		    result.enable(Capability.MISSING_VALUES);

		    // class
		    result.enable(Capability.NOMINAL_CLASS);
		    result.enable(Capability.MISSING_CLASS_VALUES);

		    // other
		    result.enable(Capability.ONLY_MULTIINSTANCE);

		    return result;
		  }
		  
		  
		  /**
		   * Get a vector of all instances in all bags.
		   * The indices in this vector are used to look up stored computations.
		   * @return
		   */
		  private ArrayList<Instance> getIndexVector(){
			  
			  ArrayList<Instance> allInstances = new ArrayList<Instance>();
			  
			  for(Instance Bag : train){			  
				  for (int i = 0; i < Bag.relationalValue(1).numInstances(); i++) {
				      Instance x = Bag.relationalValue(1).instance(i);   
				      allInstances.add(x);			  
				  }
			  }
			  
			  return allInstances;
			  
		  }
		  
		  
		  /**
		   * Determine maximum and minimum values of all instance attributes
		   * @return
		   */
		  private void precompute() {
			  
			  attrMin = new double[train.instance(0).relationalValue(1).instance(0).numAttributes()];
			  Arrays.fill(attrMin, Double.MAX_VALUE);
			  attrMax = new double[train.instance(0).relationalValue(1).instance(0).numAttributes()];
			  Arrays.fill(attrMax, Double.NEGATIVE_INFINITY);

			  for(Instance Bag : train){			  
				  for (int i = 0; i < Bag.relationalValue(1).numInstances(); i++) {
				      Instance x = Bag.relationalValue(1).instance(i);
	
				      for(int a = 0; a < x.numAttributes(); a++){
				    	  if(x.attribute(a).isNumeric()){
				    		 double val = x.value(a); 
				    		 if(val < attrMin[a]){
				    			 attrMin[a] = val;
				    		 }
				    		 if(val > attrMax[a]){
				    			 attrMax[a] = val;
				    		 }
				    	  }
				      }
				  }
			  }	
		  
			  
		  }

		
		/**
		 * Determine an OWA weight vector of the specified type and size.
		 * @param type
		 * @param size
		 * @return
		 */
		private double[] getWeights(String type, int size) {
			if(type.equals("max")){
				double[] weights = new double[size];
			    double p = size;
			    double Z = p * (p + 1);
			    for (int i = 1; i <= size; i++) {
			      weights[i - 1] = 2 * (p - i + 1) / Z;
			    }
			    return weights;
			} else if(type.equals("maxadd")){
				double[] weights = new double[size];
			    double p = size;
			    double sum = 0.0;
			    for(int i = 1; i <= p; i++){
			    	sum += (double) 1/i;
			    }
			    for (int i = 1; i <= size; i++) {
			      weights[i - 1] = 1 / (i * sum);
			    }		    
			    return weights;
			} else if(type.equals("min")){
				double[] weights = new double[size];
			    double p = size;
			    double Z = p * (p + 1);
			    for (int i = 1; i <= size; i++) {
			      weights[i - 1] = 2 * i / Z;
			    }
			    return weights;
			} else if(type.equals("minadd")){
				double[] weights = new double[size];
			    double p = size;
			    double sum = 0.0;
			    for(int i = 1; i <= p; i++){
			    	sum += (double) 1/i;
			    }
			    for (int i = 1; i <= size; i++) {
			      weights[i - 1] = 1 / ((p - i + 1) * sum);
			    }		    
			    return weights;
			} else {
				throw new IllegalArgumentException("Invalid OWA weights.");
			}
		}

	  
	  
	  
	  
	  

}
