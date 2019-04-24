package twoClasses.imbalancedFuzzyRough;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities.Capability;


/**
 * Builder for fuzzy rough instance based MIL classifiers for imbalanced classification
 *
 * @author Sarah
 */
public class BagBased extends AbstractClassifier implements Serializable{

	
	  /**
	   * Constant for the evaluation of the default merging function.
	   */
	  private double beta = 1;

	  /**
	   * The training data
	   */
	  private Instances train;

	  /** for serialization */
	  private static final long serialVersionUID = 1L;
	  
	  private String bagToClassMembership; //"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd" 
	  
	  private int nMin;
	  private int nMaj;
	  private int minorityClass;
	  private int majorityClass;
	  private double gamma;
	  private int versionIFROWANN;
	  
	  private double[][] bagToClassMembershipDegrees; // precomputed values
	  
	  
	  public BagBased(String bagToClassMembership, int versionIFROWANN, double gamma){
		  this.bagToClassMembership = bagToClassMembership;
		  this.versionIFROWANN = versionIFROWANN;
		  this.gamma = gamma;
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
	    
	    // precompute memberships of instances to class
	    bagToClassMembershipDegrees = precompute();
	    
	    // class counts
	    countClasses();
	  }

	  /**
	   * Computes the cosine similarity between two instances.
	   *
	   * @param a the first instance in the comparison.
	   * @param b the second instance in the comparison.
	   * @return number between 0 and 1 representing the cosine similarity.
	   */
	  private double cos(Instance a, Instance b) {
	    double ab = 0, a2 = 0, b2 = 0;
	    for (int i = 0; i < a.numAttributes(); i++) {
	      ab += a.value(i) * b.value(i);
	      a2 += a.value(i) * a.value(i);
	      b2 += b.value(i) * b.value(i);
	    }
	    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
	    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
	    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));
	    return (s > 1)? 1 : s;
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
	      double s = 1 - cos(x, a);
	      if (s < m)
	        m = s;
	    }
	    return m;
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

	  /**
	   * Computes the Hausdorff similarity between two bags. It uses cosine
	   * distance between instances. Is the complement of the Hausdorff distance.
	   *
	   * @param a the first bag.
	   * @param b the second bag.
	   * @return number between 0 and 1 representing the Hausdorff distance.
	   */
	  public double averageHausdorffSimilarity(Instance A, Instance B) {
	    return 1 - averageHausdorffDistance(A, B);
	  }
	  
	  /**
	   * Compute the Lukasiewicz Implication between two values.
	   * @param a first value.
	   * @param b second value.
	   * @return the Lukasiewicz Implication between a and b.
	   */
	  private double lukasiewiczImplication(double a, double b) {
	    double c = 1 - a + b;
	    return (c < 1)? c: 1;
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
	  private double lowerAppMembershipDegree(Instance X, int classLabel) {
	    ArrayList<Double> allValues = new ArrayList<Double>();
	    for (int j = 0; j < train.numInstances(); j++) {
		      Instance B = train.instance(j);
		      double R = averageHausdorffSimilarity(X, B);
		      //double M = B.classValue() == classLabel ? 1.0 : 0.0;		      
		      double M = bagToClassMembershipDegrees[j][classLabel];
		      allValues.add(lukasiewiczImplication(R, M));
		}
	    Collections.sort(allValues, Collections.reverseOrder());
	    double[] weights;
	    if(classLabel == minorityClass){
    	  weights = getWeights("imb-pos", allValues.size());
	    } else {
    	  weights = getWeights("imb-neg", allValues.size());
	    }
	    double thisValue = 0.0;
	    for(int el = 0; el < allValues.size(); el++){
    	  thisValue += weights[el] * allValues.get(el);
	    }
	    return thisValue;  
	    
	  }
	  
		public int getPosClass(){
			return minorityClass;
		}

	  /**
	   * Function used to merge upper and lower membership degrees into a single class
	   * membership degree. The default merging function is the weighted average of
	   * the membership degrees to the lower and upper class approximations. <tt>beta</tt>
	   * is the weight of the lower approximation.
	   */
	  private double merge(double lowerApp, double upperApp) {
	    return (beta * lowerApp + upperApp) / (beta + 1);
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
	    
	    // minority class
	    double lowerAppMembDegree = lowerAppMembershipDegree(exmp, minorityClass);
	    double upperAppMembDegree = 1.0 - lowerAppMembershipDegree(exmp, majorityClass);
	    distribution[minorityClass] = merge(lowerAppMembDegree, upperAppMembDegree);
	    
	    // majority class
	    lowerAppMembDegree = lowerAppMembershipDegree(exmp, majorityClass);
	    upperAppMembDegree = 1.0 - lowerAppMembershipDegree(exmp, minorityClass);
	    distribution[majorityClass] = merge(lowerAppMembDegree, upperAppMembDegree);

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
	  
	  private void countClasses() {
		  int nClasses = train.classAttribute().numValues();		  
		  int[] counts = new int[nClasses];
		  
		  for(Instance Bag : train){
			  counts[(int) Bag.classValue()]++;
		  }
		  
		  if(counts[0] < counts[1]){
			  minorityClass = 0;
			  majorityClass = 1;
		  } else {
			  minorityClass = 1;
			  majorityClass = 0;
		  }
		  
		  nMin = counts[minorityClass];
		  nMaj = counts[majorityClass];
			
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
			} else if(type.equals("imb-pos")){
				
				if(versionIFROWANN == 1 || versionIFROWANN == 2){ // W_p^l1
					
					double[] weights = new double[size];
					double n = nMaj;
					double fact = 1.0;
					double denom = (double) n * (n + 1.0);
					for(int pos = nMin; pos < size; pos++){
						weights[pos] = (double) 2.0 * fact/ denom;
						fact++;
					}
					
					return weights;				
					
				} else if(versionIFROWANN == 3 || versionIFROWANN == 4){ // W_p^l2
					double[] weights = new double[size];
					double n = nMaj;
					
					if(n < 1024){
						int exp = 0;
						for(int pos = nMin; pos < size; pos++){
							weights[pos] = (double)Math.pow(2, exp)/(Math.pow(2, n)-1);
							exp++;
						}
					} else {
						weights[size - 1] = 0.5;
						for(int pos = size - 2; pos >= nMin; pos--){
							weights[pos] = weights[pos + 1] / 2.0;
						}
					}
					
					return weights;
				} else if(versionIFROWANN == 5 || versionIFROWANN == 6){ // W_p^{l1,gamma}
					
					
					double[] weights = new double[size];
					int n = nMaj;
					int p = nMin;
					int r = (int) Math.ceil(p + gamma * (n - p));
					
					double denom = (double) r * (r + 1.0);
					double count_enum = 1;
					for(int pos = size - r; pos < size; pos++){
						weights[pos] = 2 *  count_enum / denom;
						count_enum++;
					}
					
					return weights;
					
				} else {
					throw new IllegalArgumentException("Invalid version of weight vector IFROWANN.");
				}

			} else if(type.equals("imb-neg")){ 
				
					if(versionIFROWANN == 2 || versionIFROWANN == 4 || versionIFROWANN == 6){ // W_N^l2
					
					double[] weights = new double[size];
					double p = nMin;
					
					if(p < 1024){
						int exp = 0;
						for(int pos = nMaj; pos < size; pos++){
							weights[pos] = (double)Math.pow(2, exp)/(Math.pow(2, p)-1);
							exp++;
						}
					} else {
						weights[size - 1] = 0.5;
						for(int pos = size - 2; pos >= nMaj; pos--){
							weights[pos] = weights[pos + 1] / 2.0;
						}
					}
					
					return weights;	
				} else if(versionIFROWANN == 1 || versionIFROWANN == 3 || versionIFROWANN == 5){ // W_N^l1
					
					double[] weights = new double[size];
					double p = nMin;
					double fact = 1.0;
					double denom = (double) p * (p + 1.0);
					for(int pos = nMaj; pos < size; pos++){
						weights[pos] = (double) 2.0 * fact/ denom;
						fact++;
					}
					
					return weights;	
					
				} else {
					throw new IllegalArgumentException("Invalid version of weight vector IFROWANN.");
				}
				
			} else {
				throw new IllegalArgumentException("Invalid OWA weights.");
			}
		}
		
		
		  /**
		   * Determine the membership degrees of all bags to all classes
		   * @return
		   */
		  private double[][] precompute() {
			  int nClasses = train.classAttribute().numValues();		  
			  double[][] mems = new double[train.numInstances()][nClasses];

			  for (int j = 0; j < train.numInstances(); j++) {
				  Instance B = train.instance(j);
				  
				  for(int cl = 0; cl < nClasses; cl++){
					  mems[j][cl] = bagToClass(B, cl);
				  }
			  }

			  return mems;			  
		  }
		  
		  /**
		   * Determine the membership degrees of a bag to a class
		   * @return
		   */
		  private double bagToClass(Instance Bag, int classLabel){
			  
			  if(bagToClassMembership.equals("-Ave")){
				  
				  double avg = 0;
				  int count = 0;
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;
				      avg += averageHausdorffSimilarity(Bag, B);
				      count++;
				  }
				  avg /= count;
				  return avg;
				  
			  } else if(bagToClassMembership.equals("-OWAmax")){
				  
				  ArrayList<Double> allValues = new ArrayList<Double>();
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;
				      allValues.add(averageHausdorffSimilarity(Bag, B));
				  }
				  Collections.sort(allValues, Collections.reverseOrder());
				  double[] weights = getWeights("max", allValues.size());
				  double thisValue = 0.0;
				  for(int el = 0; el < allValues.size(); el++){
			    	  thisValue += weights[el] * allValues.get(el);
				  }	    
				  return thisValue; 
				  
			  } else if(bagToClassMembership.equals("-CompAve")){
				  
				  double avg = 0;
				  int count = 0;
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;
				      avg += averageHausdorffSimilarity(Bag, B);
				      count++;
				  }
				  avg /= count;
				  return 1.0 - avg;
				  
			  } else if(bagToClassMembership.equals("-CompOWAmin")){
				  
				  ArrayList<Double> allValues = new ArrayList<Double>();
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;
				      allValues.add(averageHausdorffSimilarity(Bag, B));
				  }
				  Collections.sort(allValues, Collections.reverseOrder());
				  double[] weights = getWeights("min", allValues.size());
				  double thisValue = 0.0;
				  for(int el = 0; el < allValues.size(); el++){
			    	  thisValue += weights[el] * allValues.get(el);
				  }	    
				  return 1.0 - thisValue; 
				  
			  } else if(bagToClassMembership.equals("-OWAmaxAdd")){
				  
				  ArrayList<Double> allValues = new ArrayList<Double>();
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() != classLabel) continue;
				      allValues.add(averageHausdorffSimilarity(Bag, B));
				  }
				  Collections.sort(allValues, Collections.reverseOrder());
				  double[] weights = getWeights("maxadd", allValues.size());
				  double thisValue = 0.0;
				  for(int el = 0; el < allValues.size(); el++){
			    	  thisValue += weights[el] * allValues.get(el);
				  }	    
				  return thisValue; 
				  
			  } else if(bagToClassMembership.equals("-CompOWAminAdd" )){
				  
				  ArrayList<Double> allValues = new ArrayList<Double>();
				  for (int j = 0; j < train.numInstances(); j++) {
				      Instance B = train.instance(j);
				      if (B.classValue() == classLabel) continue;
				      allValues.add(averageHausdorffSimilarity(Bag, B));
				  }
				  Collections.sort(allValues, Collections.reverseOrder());
				  double[] weights = getWeights("minadd", allValues.size());
				  double thisValue = 0.0;
				  for(int el = 0; el < allValues.size(); el++){
			    	  thisValue += weights[el] * allValues.get(el);
				  }	    
				  return 1.0 - thisValue; 
				  
			  } else {
				  throw new IllegalArgumentException("Invalid bag-to-class membership."); 
			  }
		  }
		  
	  
}
