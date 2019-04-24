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

public class TFBMFactory_Sarah extends AbstractClassifier implements Serializable{
	

	  /**
	   * The training data
	   */
	  private Instances train;

	  /** for serialization */
	  private static final long serialVersionUID = 1L;
	  
	  private String bagSimilarity; // "-Haus", "-HausL", "-HausIA", "-AvgH", "-AvgHL", "-AvgIA"
	  private String bagToClassMembership; //"-Avg", "-OWAL", "-OWAIA", "-CAvg", "-CAvgL", "-CAvgIA" 
	  
	  
	  private double[] attrMin;
	  private double[] attrMax;
	  
	  
	  public TFBMFactory_Sarah(String bagSimilarity, String bagToClassMembership){
		  this.bagSimilarity = bagSimilarity;
		  this.bagToClassMembership = bagToClassMembership;		  
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
	    
	    // precompute attribute ranges for normalization in Euclidean distance
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
	   * Computes the minimal distance between an instances and a bag. It uses cosine
	   * distance between instances.
	   *
	   * @param a the instance.
	   * @param b the bag.
	   * @return number between 0 and 1 representing the minimal distance.
	   */
	  private double min(Instance a, Instance B) {
	    double m = Double.MAX_VALUE;
	    Instances bagB = B.relationalValue(1);
	    for (int i = 0; i < bagB.numInstances(); i++) {
	      Instance x = bagB.instance(i);
	      double s = 1 - instanceSimilarity(x, a);
	      if (s < m)
	        m = s;
	    }
	    return m;
	  }
	  
	  private double min_OWA(Instance a, Instance B, String whichOWA) {
		    ArrayList<Double> allValues = new ArrayList<Double>();
		    Instances bagB = B.relationalValue(1);
		    for (int i = 0; i < bagB.numInstances(); i++) {
		      Instance x = bagB.instance(i);
		      double s = 1 - instanceSimilarity(x, a);
		      allValues.add(s);
		    }
		    Collections.sort(allValues, Collections.reverseOrder());
		    double[] weights = whichOWA.equals("lin") ? getWeights("min", allValues.size()) : getWeights("minadd", allValues.size())  ;
		    double thisValue = 0.0;
		    for(int el = 0; el < allValues.size(); el++){
	    	  thisValue += weights[el] * allValues.get(el);
		    }
		    return thisValue; 
		  }

	  /**
	   * Computes the Hausdorff hemi-distance between two bags. It uses cosine
	   * distance between instances.
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the Hausdorff hemi-distance.
	   */
	  private double sum_min(Instance A, Instance B) {
	    double sum = 0;
	    Instances bagA = A.relationalValue(1);
	    for (int i = 0; i < bagA.numInstances(); i++) {
	      Instance x = bagA.instance(i);
	      sum += min(x, B);
	    }
	    return sum;
	  }
	  
	  private double sum_min_OWA(Instance A, Instance B, String whichOWA) {
		    double sum = 0;
		    Instances bagA = A.relationalValue(1);
		    for (int i = 0; i < bagA.numInstances(); i++) {
		      Instance x = bagA.instance(i);
		      sum += min_OWA(x, B, whichOWA);
		    }
		    return sum;
	  }
	  
	  private double max_min(Instance A, Instance B) {
		    double max = 0;
		    Instances bagA = A.relationalValue(1);
		    for (int i = 0; i < bagA.numInstances(); i++) {
		      Instance x = bagA.instance(i);
		      double s = min(x, B);
		      if (s > max){
		    	  max = s;
		      }
		    }
		    return max;
	  }
	  
	  private double max_min_OWA(Instance A, Instance B, String whichOWA) {
		    ArrayList<Double> allValues = new ArrayList<Double>();
		    Instances bagA = A.relationalValue(1);
		    for (int i = 0; i < bagA.numInstances(); i++) {
		      Instance x = bagA.instance(i);
		      double s = min_OWA(x, B, whichOWA);
		      allValues.add(s);
		    }
		    Collections.sort(allValues, Collections.reverseOrder());
		    double[] weights = whichOWA.equals("lin") ? getWeights("max", allValues.size()) : getWeights("maxadd", allValues.size())  ;
		    double thisValue = 0.0;
		    for(int el = 0; el < allValues.size(); el++){
	    	  thisValue += weights[el] * allValues.get(el);
		    }
		    return thisValue; 
	  }

	  /**
	   * Computes the Hausdorff distance between two bags. It uses cosine
	   * distance between instances.
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the Hausdorff distance.
	   */
	  public double averageHausdorffDistance(Instance A, Instance B) {
	    double n1 = sum_min(A, B);
	    double n2 = sum_min(B, A);
	    return (n1 + n2) / (A.relationalValue(1).numInstances() + B.relationalValue(1).numInstances());
	  }
	  
	  
	  public double hausdorffDistance(Instance A, Instance B) {
		    double n1 = max_min(A, B);
		    double n2 = max_min(B, A);
		    return Math.max(n1, n2);
	  }
	  
	  public double hausdorffDistanceOWA(Instance A, Instance B, String whichOWA) {
		    double n1 = max_min_OWA(A, B, whichOWA);
		    double n2 = max_min_OWA(B, A, whichOWA);
		    return Math.max(n1, n2);
	  }
	  
	  
	  
	  /**
	   * Computes the Hausdorff distance between two bags. It uses cosine
	   * distance between instances.
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the Hausdorff distance.
	   */
	  public double averageHausdorffDistanceOWA(Instance A, Instance B, String whichOWA) {
	    double n1 = sum_min_OWA(A, B, whichOWA);
	    double n2 = sum_min_OWA(B, A, whichOWA);
	    return (n1 + n2) / (A.relationalValue(1).numInstances() + B.relationalValue(1).numInstances());
	  }
	  
	  
	  

	  /**
	   * Computes the Hausdorff similarity between two bags.
	   * Is the complement of the Hausdorff distance.
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the Hausdorff distance.
	   */
	  public double averageHausdorffSimilarity(Instance A, Instance B) {
	    return 1 - averageHausdorffDistance(A, B);
	  }
	  
	  
	  /**
	   * Computes the similarity between two bags. 
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the similarity.
	   */
	  public double similarityBetweenBags(Instance A, Instance B) {
		  
		  if(bagSimilarity.equals("-Haus")){
			  return 1 - hausdorffDistance(A,B);
		  } else if(bagSimilarity.equals("-HausL")){
			  return 1 - hausdorffDistanceOWA(A,B, "lin");
		  } else if(bagSimilarity.equals("-AvgH")){
			  return 1 - averageHausdorffDistance(A, B);			  
		  } else if(bagSimilarity.equals("-AvgHL")){
			  return 1 - averageHausdorffDistanceOWA(A, B, "lin");
		  } else if(bagSimilarity.equals("-HausIA")){
			  return 1 - hausdorffDistanceOWA(A,B, "add");
		  } else if(bagSimilarity.equals("-AvgIA")){
			  return 1 - averageHausdorffDistanceOWA(A, B, "add");
		  } else {
			  throw new IllegalArgumentException("Invalid bag-wise similarity.");
		  }	    
	  }	  

	  
	  
	  /**
	   * Computes the membership degree of a bag X to the lower approximation of
	   * the given class.
	   *
	   * @param X the bag.
	   * @param classLabel label of the given class.
	   * @return the membership degree of a bag X to the lower approximation of
	   * the given class.
	   */
	  private double membership(Instance X, int classLabel) {
		  
		  if(bagToClassMembership.equals("-Avg")){
			  
				double sum = 0;
				int count = 0;
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;  // BagsFromClass
				      count++;
				      sum += similarityBetweenBags(X,B);
				}
			    return sum/count;  
			    
		  } else if(bagToClassMembership.equals("-OWAL")){
			  
			    ArrayList<Double> allValues = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;  // BagsFromClass
				      allValues.add(similarityBetweenBags(X,B));
				}
			    Collections.sort(allValues, Collections.reverseOrder());
			    double[] weights = getWeights("max", allValues.size());

			    double thisValue = 0.0;
			    for(int el = 0; el < allValues.size(); el++){
		    	  thisValue += weights[el] * allValues.get(el);
			    }
			    return thisValue; 
			    
		  } else if(bagToClassMembership.equals("-OWAIA")){
			  
			    ArrayList<Double> allValues = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;  // BagsFromClass
				      allValues.add(similarityBetweenBags(X,B));
				}
			    Collections.sort(allValues, Collections.reverseOrder());
			    double[] weights = getWeights("maxadd", allValues.size());

			    double thisValue = 0.0;
			    for(int el = 0; el < allValues.size(); el++){
		    	  thisValue += weights[el] * allValues.get(el);
			    }
			    return thisValue; 
			    
		  } else if(bagToClassMembership.equals("-CAvg")){
			  
			    double sum = 0;
				int count = 0;
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;  // BagsNotFromClass
				      count++;
				      sum += similarityBetweenBags(X,B);
				}
			    return 1.0 - sum/count;

		  } else if(bagToClassMembership.equals("-CAvgL")){
			  
			    ArrayList<Double> allValues = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;  // BagsNotFromClass
				      allValues.add(similarityBetweenBags(X,B));
				}
			    Collections.sort(allValues, Collections.reverseOrder());
			    double[] weights = getWeights("min", allValues.size());

			    double thisValue = 0.0;
			    for(int el = 0; el < allValues.size(); el++){
		    	  thisValue += weights[el] * allValues.get(el);
			    }
			    return 1.0 - thisValue; 
			  
		  } else if(bagToClassMembership.equals("-CAvgIA")){
			    
			    ArrayList<Double> allValues = new ArrayList<Double>();
			    for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;  // BagsNotFromClass
				      allValues.add(similarityBetweenBags(X,B));
				}
			    Collections.sort(allValues, Collections.reverseOrder());
			    double[] weights = getWeights("minadd", allValues.size());

			    double thisValue = 0.0;
			    for(int el = 0; el < allValues.size(); el++){
		    	  thisValue += weights[el] * allValues.get(el);
			    }
			    return 1.0 - thisValue; 
			  
		  } else {
			  throw new IllegalArgumentException("Invalid bag-to-class-app membership."); 
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
			}  else {
				throw new IllegalArgumentException("Invalid OWA weights.");
			}
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
		  
		  
	  

}
