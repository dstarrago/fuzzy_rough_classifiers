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
 * Builder for fuzzy rough instance based MIL classifiers for imbalanced classification.
 *
 * @author Sarah
 */
public class InstanceBased extends AbstractClassifier implements Serializable {
	
	private String bagToClassAppMembership;      // "-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"
	private String instanceToClassMembership;    // "-Ave", "-OWAmax", "-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"
	private String instanceToBagMembership;      // "-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"
    private double beta;
    private int versionIFROWANN;
    
    private ArrayList<Instance> indexOfInstances; // look-up vector
    private double[][] instanceToClassMembershipDegrees; // precomputed values
    
    private int nMin;
    private int nMaj;
    private int minorityClass;
    private int majorityClass;
    private double gamma;

	  /**
	   * The training data
	   */
	  private Instances train;
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;
	  
	  public InstanceBased(String bagToClassAppMembership, String instanceToClassMembership,
			  String instanceToBagMembership, double beta, int versionIFROWANN, double gamma){
		  this.bagToClassAppMembership = bagToClassAppMembership;
		  this.instanceToClassMembership = instanceToClassMembership;
		  this.instanceToBagMembership = instanceToBagMembership;
		  this.beta = beta;
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
	    indexOfInstances = getIndexVector();
	    instanceToClassMembershipDegrees = precompute();
	    
	    // class counts
	    countClasses();
	    
	  }
	  
	public int getPosClass(){
		return minorityClass;
	}



	private void countClasses() {
		  int nClasses = train.classAttribute().numValues();		  
		  int[] counts = new int[nClasses];
		  
		  for(Instance Bag : train){
			  counts[(int) Bag.classValue()] += Bag.relationalValue(1).numInstances();
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
		  if(instanceToClassMembership.equals("-Ave")){			  
			  
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
			        double R = cos(x, y);
			        if (R > memToBag)
			        	memToBag = R;
			      }
			      
		      } else if(instanceToBagMembership.equals("-OWAmax")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
			      }
			      
			      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
			      double[] weights = getWeights("max", allValues.size());
			      memToBag = 0.0;
			      for(int i = 0; i < allValues.size(); i++){
			    	  memToBag += weights[i] * allValues.get(i);
			      }
		    	  
		      } else if(instanceToBagMembership.equals("-Ave")){
		    	  
		    	  memToBag = 0.0;
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        memToBag += cos(x, y);
			      }
			      memToBag /= B.relationalValue(1).numInstances();
		    	  
		      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
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
		  else if(instanceToClassMembership.equals("-OWAmax")){			  
			  
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
			        double R = cos(x, y);
			        if (R > memToBag)
			        	memToBag = R;
			      }
			      
		      } else if(instanceToBagMembership.equals("-OWAmax")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
			      }
			      
			      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
			      double[] weights = getWeights("max", allValues.size());
			      memToBag = 0.0;
			      for(int i = 0; i < allValues.size(); i++){
			    	  memToBag += weights[i] * allValues.get(i);
			      }
		    	  
		      } else if(instanceToBagMembership.equals("-Ave")){
		    	  
		    	  memToBag = 0.0;
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        memToBag += cos(x, y);
			      }
			      memToBag /= B.relationalValue(1).numInstances();
		    	  
		      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
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
		  else if(instanceToClassMembership.equals("-CompAve")){ 
  
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
			        double R = cos(x, y);
			        if (R > memToBag)
			        	memToBag = R;
			      }
			      
		      } else if(instanceToBagMembership.equals("-OWAmax")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
			      }
			      
			      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
			      double[] weights = getWeights("max", allValues.size());
			      memToBag = 0.0;
			      for(int i = 0; i < allValues.size(); i++){
			    	  memToBag += weights[i] * allValues.get(i);
			      }
		    	  
		      } else if(instanceToBagMembership.equals("-Ave")){
		    	  
		    	  memToBag = 0.0;
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        memToBag += cos(x, y);
			      }
			      memToBag /= B.relationalValue(1).numInstances();
		    	  
		      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
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
		  else if(instanceToClassMembership.equals("-CompOWAmin")){
			  
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
			        double R = cos(x, y);
			        if (R > memToBag)
			        	memToBag = R;
			      }
			      
		      } else if(instanceToBagMembership.equals("-OWAmax")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
			      }
			      
			      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
			      double[] weights = getWeights("max", allValues.size());
			      memToBag = 0.0;
			      for(int i = 0; i < allValues.size(); i++){
			    	  memToBag += weights[i] * allValues.get(i);
			      }
		    	  
		      } else if(instanceToBagMembership.equals("-Ave")){
		    	  
		    	  memToBag = 0.0;
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        memToBag += cos(x, y);
			      }
			      memToBag /= B.relationalValue(1).numInstances();
		    	  
		      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
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
		  else if(instanceToClassMembership.equals("-OWAmaxAdd")){
			  
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
			        double R = cos(x, y);
			        if (R > memToBag)
			        	memToBag = R;
			      }
			      
		      } else if(instanceToBagMembership.equals("-OWAmax")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
			      }
			      
			      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
			      double[] weights = getWeights("max", allValues.size());
			      memToBag = 0.0;
			      for(int i = 0; i < allValues.size(); i++){
			    	  memToBag += weights[i] * allValues.get(i);
			      }
		    	  
		      } else if(instanceToBagMembership.equals("-Ave")){
		    	  
		    	  memToBag = 0.0;
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        memToBag += cos(x, y);
			      }
			      memToBag /= B.relationalValue(1).numInstances();
		    	  
		      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
		    	  
		    	  ArrayList<Double> allValues = new ArrayList<Double>();
		    	  
			      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
			        Instance y = B.relationalValue(1).instance(k);
			        allValues.add(cos(x, y));
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
		  else if(instanceToClassMembership.equals("-CompOWAminAdd")){
			  
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
				        double R = cos(x, y);
				        if (R > memToBag)
				        	memToBag = R;
				      }
				      
			      } else if(instanceToBagMembership.equals("-OWAmax")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(cos(x, y));
				      }
				      
				      Collections.sort(allValues, Collections.reverseOrder()); // decreasing
				      double[] weights = getWeights("max", allValues.size());
				      memToBag = 0.0;
				      for(int i = 0; i < allValues.size(); i++){
				    	  memToBag += weights[i] * allValues.get(i);
				      }
			    	  
			      } else if(instanceToBagMembership.equals("-Ave")){
			    	  
			    	  memToBag = 0.0;
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        memToBag += cos(x, y);
				      }
				      memToBag /= B.relationalValue(1).numInstances();
			    	  
			      } else if(instanceToBagMembership.equals("-OWAmaxAdd")){
			    	  
			    	  ArrayList<Double> allValues = new ArrayList<Double>();
			    	  
				      for (int k = 0; k < B.relationalValue(1).numInstances(); k++) {
				        Instance y = B.relationalValue(1).instance(k);
				        allValues.add(cos(x, y));
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
	      /////////////////////////
		  //////// Maximum ////////
          /////////////////////////
		  if(bagToClassAppMembership.equals("-Max")){
	    
			    double max = 0;
			    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
			      Instance x = X.relationalValue(1).instance(i);

			      ArrayList<Double> allValues = new ArrayList<Double>();
		    	  int index = 0;
			      for (int j = 0; j < train.numInstances(); j++) {
			        Instance B = train.instance(j);
			        for (int k = 0; k < B.relationalValue(1).numInstances(); k++) { 				        	
			          Instance y = B.relationalValue(1).instance(k);
			          double R = cos(x, y);
			          double M = instanceToClassMembershipDegrees[index][classLabel];// Sarah
			          allValues.add(lukasiewiczImplication(R, M));
			          index++;
			        }
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
			      
			      if (thisValue > max)
			        max = thisValue;
			    }
			    return max;
			    			    
		  } 
	      /////////////////////////////
		  //////// OWA-Maximum ////////
          /////////////////////////////		  
		  else if(bagToClassAppMembership.equals("-OWAmax")){			  
			  
			    ArrayList<Double> values = new ArrayList<Double>();
			    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
			      Instance x = X.relationalValue(1).instance(i);
			      
			      ArrayList<Double> allValues = new ArrayList<Double>();
		    	  int index = 0;
			      for (int j = 0; j < train.numInstances(); j++) {
			        Instance B = train.instance(j);
			        for (int k = 0; k < B.relationalValue(1).numInstances(); k++) { 				        	
			          Instance y = B.relationalValue(1).instance(k);
			          double R = cos(x, y);
			          double M = instanceToClassMembershipDegrees[index][classLabel];// Sarah
			          allValues.add(lukasiewiczImplication(R, M));
			          index++;
			        }
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
			      
			      values.add(thisValue);
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
		  else if(bagToClassAppMembership.equals("-Ave")){

			    double avg = 0;
			    int count = 0;
			    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
			      Instance x = X.relationalValue(1).instance(i);
			      
			      ArrayList<Double> allValues = new ArrayList<Double>();
		    	  int index = 0;
			      for (int j = 0; j < train.numInstances(); j++) {
			        Instance B = train.instance(j);
			        for (int k = 0; k < B.relationalValue(1).numInstances(); k++) { 				        	
			          Instance y = B.relationalValue(1).instance(k);
			          double R = cos(x, y);
			          double M = instanceToClassMembershipDegrees[index][classLabel];// Sarah
			          allValues.add(lukasiewiczImplication(R, M));
			          index++;
			        }
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
			      
			      avg += thisValue;
			      count++;
			    }
			    return (double) avg / count;			  

		  } 
	      ////////////////////////////
		  //////// OWA-MaxAdd ////////
          ////////////////////////////	
		  else if(bagToClassAppMembership.equals("-OWAmaxAdd")){
			  
			    ArrayList<Double> values = new ArrayList<Double>();
			    for (int i = 0; i < X.relationalValue(1).numInstances(); i++) {
			      Instance x = X.relationalValue(1).instance(i);
			      
			      ArrayList<Double> allValues = new ArrayList<Double>();
		    	  int index = 0;
			      for (int j = 0; j < train.numInstances(); j++) {
			        Instance B = train.instance(j);
			        for (int k = 0; k < B.relationalValue(1).numInstances(); k++) { 				        	
			          Instance y = B.relationalValue(1).instance(k);
			          double R = cos(x, y);
			          double M = instanceToClassMembershipDegrees[index][classLabel];// Sarah
			          allValues.add(lukasiewiczImplication(R, M));
			          index++;
			        }
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

			      values.add(thisValue);
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
	   * Determine the membership degrees of all instances to all classes
	   * @return
	   */
	  private double[][] precompute() {
		  int nClasses = train.classAttribute().numValues();		  
		  double[][] mems = new double[indexOfInstances.size()][nClasses];
		  
		  int index = 0;
		  for(Instance Bag : train){			  
			  for (int i = 0; i < Bag.relationalValue(1).numInstances(); i++) {
			      Instance x = Bag.relationalValue(1).instance(i);
			      //int index = findIndex(x);		
			      
			      for(int cl = 0; cl < nClasses; cl++){
			    	  mems[index][cl] = instanceToClassMembership(x, cl);
			      }
			      index++;
			  }
		  }
		  
		  return mems;
		  
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

}
