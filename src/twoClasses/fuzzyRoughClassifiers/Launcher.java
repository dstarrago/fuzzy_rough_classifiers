/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package twoClasses.fuzzyRoughClassifiers;

import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

/**
 * Class for testing the classifiers.
 *
 * @author Danel
 */
public class Launcher {

  //private final String dataName = "Musk1";
  //private final String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/"
  //        + "_Colección MIL #1/01 Musk1/";
  private String dataName = "Musk1"; // Sarah
  private String path = "/home/sarah/Data/01 Musk1/"; // Sarah
  private final int folds = 5;
  private final String outputDir = String.format("%d-Folds-CV/", folds);
  private final int posClassLabel = 1;
  
//set up to use all datasets
 private String[] dataNames;
 private String[] paths;
 private int[] posClasses;
 
 
 private void setDatasets(){
	  dataNames = new String[42];
	  paths = new String[42];
	  
	  // Musk1
	  dataNames[0] = "Musk1";
      paths[0] = "/home/sarah/Data/01 Musk1/";
	  // Musk2
	  dataNames[1] = "Musk2";
      paths[1] = "/home/sarah/Data/02 Musk2/";	  
	  // Atoms
	  dataNames[2] = "Atoms";
      paths[2] = "/home/sarah/Data/03 Atoms/";	  
	  // Bonds
	  dataNames[3] = "Bonds";
      paths[3] = "/home/sarah/Data/04 Bonds/";	  
	  // Chains
	  dataNames[4] = "Chains";
      paths[4] = "/home/sarah/Data/05 Chains/";	  
//	  // Suramin
//	  dataNames[5] = "Suramin";
//      paths[5] = "/home/sarah/Data/06 Suramin/";	  
	  // Elephant
	  dataNames[6] = "Elephant";
      paths[6] = "/home/sarah/Data/08 Elephant/";	  
	  // Fox
	  dataNames[7] = "Fox";
      paths[7] = "/home/sarah/Data/09 Fox/";	  
	  // Tiger
	  dataNames[8] = "Tiger";
      paths[8] = "/home/sarah/Data/10 Tiger/";	  
	  // EastWest
	  dataNames[9] = "EastWest";
      paths[9] = "/home/sarah/Data/11 EastWest/";	  
	  // WestEast
	  dataNames[10] = "WestEast";
      paths[10] = "/home/sarah/Data/12 WestEast/";	  
	  // AntDrugs5
	  dataNames[11] = "AntDrugs5";
      paths[11] = "/home/sarah/Data/16 AntDrugs5/";	  
	  // AntDrugs10
	  dataNames[12] = "AntDrugs10";
      paths[12] = "/home/sarah/Data/17 AntDrugs10/";	  
	  // AntDrugs20
	  dataNames[13] = "AntDrugs20";
      paths[13] = "/home/sarah/Data/18 AntDrugs20/";	  
	  // TREC9-1 
	  dataNames[14] = "TREC9Sel-1";
      paths[14] = "/home/sarah/Data/19 TREC9-1/";	  
	  // TREC9-2
	  dataNames[15] = "TREC9Sel-2";
      paths[15] = "/home/sarah/Data/20 TREC9-2/";	  
	  // TREC9-3
	  dataNames[16] = "TREC9Sel-3";
      paths[16] = "/home/sarah/Data/21 TREC9-3/";	  
	  // TREC9-4
	  dataNames[17] = "TREC9Sel-4";
      paths[17] = "/home/sarah/Data/22 TREC9-4/";	  
	  // TREC9-7
	  dataNames[18] = "TREC9Sel-7";
      paths[18] = "/home/sarah/Data/23 TREC9-7/";	  
	  // TREC9-9
	  dataNames[19] = "TREC9Sel-9";
      paths[19] = "/home/sarah/Data/24 TREC9-9/";	  
	  // TREC9-10
	  dataNames[20] = "TREC9Sel-10";
      paths[20] = "/home/sarah/Data/25 TREC9-10/";	  
	  // WIR7
	  dataNames[21] = "WIRSel-7";
      paths[21] = "/home/sarah/Data/32 WIR7/";	  
	  // WIR8
	  dataNames[22] = "WIRSel-8";
      paths[22] = "/home/sarah/Data/33 WIR8/";	  
	  // WIR9
	  dataNames[23] = "WIRSel-9";
      paths[23] = "/home/sarah/Data/34 WIR9/";	  
	  // CLJ-16.30.2
	  dataNames[24] = "CLJ-16.30.2";
      paths[24] = "/home/sarah/Data/55 CLJ-16.30.2/";	  
	  // CLJ-16-50-2
	  dataNames[25] = "CLJ-16-50-2";
      paths[25] = "/home/sarah/Data/56 CLJ-16-50-2/";	  
	  // CLJ-80.166.1
	  dataNames[26] = "CLJ-80.166.1";
      paths[26] = "/home/sarah/Data/57 CLJ-80.166.1/";	  
	  // CLJ-80.166.1-Strong
	  dataNames[27] = "CLJ-80.166.1-Strong";
      paths[27] = "/home/sarah/Data/58 CLJ-80.166.1-Strong/";	  
	  // CLJ-80-206-1
	  dataNames[28] = "CLJ-80-206-1";
      paths[28] = "/home/sarah/Data/59 CLJ-80-206-1/";	  
	  // CLJ-160.166.1
	  dataNames[29] = "CLJ-160.166.1";
      paths[29] = "/home/sarah/Data/60 CLJ-160.166.1/";	  
	  // CLJ-160.166.1-Strong
	  dataNames[30] = "CLJ-160.166.1-Strong";
      paths[30] = "/home/sarah/Data/61 CLJ-160.166.1-Strong/";	  
	  // CLJ-160-566-1
	  dataNames[31] = "CLJ-160-566-1";
      paths[31] = "/home/sarah/Data/62 CLJ-160-566-1/";	  
	  // Corel01vs02
	  dataNames[32] = "Corel01vs02";
      paths[32] = "/home/sarah/Data/63 Corel01vs02/";	  
	  // Corel01vs03
	  dataNames[33] = "Corel01vs03";
      paths[33] = "/home/sarah/Data/64 Corel01vs03/";	  
	  // Corel01vs04
	  dataNames[34] = "Corel01vs04";
      paths[34] = "/home/sarah/Data/65 Corel01vs04/";	  
	  // Corel01vs05
	  dataNames[35] = "Corel01vs05";
      paths[35] = "/home/sarah/Data/66 Corel01vs05/";	  
	  // Corel02vs03
	  dataNames[36] = "Corel02vs03";
      paths[36] = "/home/sarah/Data/67 Corel02vs03/";	  
	  // Corel02vs04
	  dataNames[37] = "Corel02vs04";
      paths[37] = "/home/sarah/Data/68 Corel02vs04/";	  
	  // Corel02vs05
	  dataNames[38] = "Corel02vs05";
      paths[38] = "/home/sarah/Data/69 Corel02vs05/";	  
	  // Corel03vs04
	  dataNames[39] = "Corel03vs04";
      paths[39] = "/home/sarah/Data/70 Corel03vs04/";	  
	  // Corel03vs05
	  dataNames[40] = "Corel03vs05";
      paths[40] = "/home/sarah/Data/71 Corel03vs05/";	  
      // Corel04vs05
      dataNames[41] = "Corel04vs05";
      paths[41] = "/home/sarah/Data/72 Corel04vs05/";
 }
 
 private void setDatasetsImbalanced(){
	 dataNames = new String[36];
	  paths = new String[36];
	  posClasses = new int[36];
	  Arrays.fill(posClasses,1);
	  
	  // Thioredoxin
	  dataNames[0] = "Thioredoxin";
     paths[0] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/07 Thioredoxin/";
     // Function
     dataNames[1] = "Function";
     paths[1] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/14 Function/";
	  // Process
     dataNames[2] = "Process";
     paths[2] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/13 Process/";
     // Component
     dataNames[3] = "Component";
     paths[3] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/15 Component/";
     // WIRSel-1
     dataNames[4] = "WIRSel-1";
     paths[4] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/26 WIR1/";
     // WIRSel-2
     dataNames[5] = "WIRSel-2";
     paths[5] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/27 WIR2/";
     // WIRSel-3
     dataNames[6] = "WIRSel-3";
     paths[6] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/28 WIR3/";
     // WIRSel-4
     dataNames[7] = "WIRSel-4";
     paths[7] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/29 WIR4/";
     posClasses[7] = 0;
     // WIRSel-5
     dataNames[8] = "WIRSel-5";
     paths[8] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/30 WIR5/";
     posClasses[8] = 0;
     // WIRSel-6
     dataNames[9] = "WIRSel-6";
     paths[9] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/31 WIR6/";
     posClasses[9] = 0;
	  // Corel20-1
     dataNames[10] = "Corel20-1";
     paths[10] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/35 Corel1/";
     // Corel20-2
     dataNames[11] = "Corel20-2";
     paths[11] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/36 Corel2/";
     // Corel20-3
     dataNames[12] = "Corel20-3";
     paths[12] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/37 Corel3/";
     // Corel20-4
     dataNames[13] = "Corel20-4";
     paths[13] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/38 Corel4/";
     // Corel20-5
     dataNames[14] = "Corel20-5";
     paths[14] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/39 Corel5/";
     // Corel20-6
     dataNames[15] = "Corel20-6";
     paths[15] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/40 Corel6/";
     // Corel20-7
     dataNames[16] = "Corel20-7";
     paths[16] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/41 Corel7/";
     // Corel20-8
     dataNames[17] = "Corel20-8";
     paths[17] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/42 Corel8/";
     // Corel20-9
     dataNames[18] = "Corel20-9";
     paths[18] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/43 Corel9/";
	  // Corel20-10 
     dataNames[19] = "Corel20-10";
     paths[19] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/44 Corel10/";
     // Corel20-11
     dataNames[20] = "Corel20-11";
     paths[20] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/45 Corel11/";
     // Corel20-12
     dataNames[21] = "Corel20-12";
     paths[21] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/46 Corel12/";
     // Corel20-13
     dataNames[22] = "Corel20-13";
     paths[22] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/47 Corel13/";
     // Corel20-14
     dataNames[23] = "Corel20-14";
     paths[23] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/48 Corel14/";
     // Corel20-15
     dataNames[24] = "Corel20-15";
     paths[24] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/49 Corel15/";
     // Corel20-16
     dataNames[25] = "Corel20-16";
     paths[25] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/50 Corel16/";
     // Corel20-17
     dataNames[26] = "Corel20-17";
     paths[26] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/51 Corel17/";
     // Corel20-18
     dataNames[27] = "Corel20-18";
     paths[27] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/52 Corel18/";
	  // Corel20-19
     dataNames[28] = "Corel20-19";
     paths[28] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/53 Corel19/";
     // Corel20-20
     dataNames[29] = "Corel20-20";
     paths[29] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/54 Corel20/";    
     
     // Elephant
     dataNames[30] = "Elephant";
     paths[30] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Elephant/"; 
     // Fox
     dataNames[31] = "Fox";
     paths[31] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Fox/";  
     // Mutagenesis_atoms
     dataNames[32] = "Mutagenesis_atoms";
     paths[32] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Mutagenesis_atoms/";  
     // Mutagenesis_bonds
     dataNames[33] = "Mutagenesis_bonds";
     paths[33] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Mutagenesis_bonds/";  
     // Mutagenesis_chains
     dataNames[34] = "Mutagenesis_chains";
     paths[34] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Mutagenesis_chains/";  
     // Tiger
     dataNames[35] = "Tiger";
     paths[35] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/Tiger/";
     
}
 

  private void resetWeights(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(1);
    }
  }

  public void testClassifier_5CV() {
    //FRBMx1 c = new FRBMx1();
	FRIM1 c = new FRIM1();
    System.out.println("Data " + dataName);
    System.out.println("Classifier " + c.getClass().getName());
    try {
      Evaluation eval = null;
      for (int fold = 1; fold <= folds; fold++) {
        String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
        String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
        System.out.println(trainFileName);
        Instances trainData = DataSource.read(trainFileName);
        Instances testData = DataSource.read(testFileName);
        resetWeights(trainData);
        resetWeights(testData);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
        c.buildClassifier(trainData);
        System.out.println(String.format("Built on fold %d!", fold));
        if (eval == null)
          eval = new Evaluation(trainData);
        else
          eval.setPriors(trainData);
        eval.evaluateModel(c, testData);
        System.out.println(String.format("Evaluation on fold %d done!", fold));
      }
      System.out.println();
      System.out.println("***** Micro-Average Performance *****");
      System.out.println();
      System.out.println("Accuracy " + eval.pctCorrect());
      System.out.println("Kappa " + eval.kappa());
      System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
      System.out.println("Precision " + eval.precision(posClassLabel));
      System.out.println("Recall " + eval.recall(posClassLabel));
      System.out.println("F1 " + eval.fMeasure(posClassLabel));
      System.out.println();
      System.out.println(eval.toMatrixString());
      System.out.println();
      System.out.println();
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  
  //Sarah
  public void testClassifierFamilyTFRIM_Sarah() {
	    setDatasets();
	    TFRIMFactory family = new TFRIMFactory();
	    for (int i = 0; i < family.size(); i++) {
	      FRMClassifier c = family.buildClassifier(i);
	      System.out.println((i + 1) + " De la familia: " + c.name());
	      
	      
	      for(int d = 0; d < dataNames.length; d++){
	      
//	    	  if(d != 5 && d != 1 && d != 3 && d != 4 && d != 11 && d != 12
//	    			  && d != 13
//	    					  && d != 14
//	    							  && d != 15
//	    									  && d != 16
//	    											  && d != 17
//	    													  && d != 18
//	    															  && d != 19
//	    																	  && d != 20
//	    																			  && d != 21
//	    																					  && d != 22
//	    															  && d != 23){ // ignore Suramin
	    	  if(d!=5){
		    	  dataName = dataNames[d];
		    	  path = paths[d];
		    	  
		    	  System.out.println("Dataset: " + dataName);
		    	  
			      try {
			          Evaluation eval = null;
			          for (int fold = 1; fold <= folds; fold++) {
			            String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
			            String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
			            Instances trainData = DataSource.read(trainFileName);
			            Instances testData = DataSource.read(testFileName);
			            resetWeights(trainData);
			            resetWeights(testData);
			            trainData.setClassIndex(trainData.numAttributes() - 1);
			            testData.setClassIndex(testData.numAttributes() - 1);
			            c.buildClassifier(trainData);
			            System.out.println(String.format("Built on fold %d!", fold));
			            if (eval == null)
			              eval = new Evaluation(trainData);
			            else
			              eval.setPriors(trainData);
			            eval.evaluateModel(c, testData);
			            System.out.println(String.format("Evaluation on fold %d done!", fold));
			          }
			          System.out.println();
			          System.out.println("***** Micro-Average Performance *****");
			          System.out.println();
			          System.out.println("Accuracy " + eval.pctCorrect());
			          System.out.println("Kappa " + eval.kappa());
			          System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
			          System.out.println("Precision " + eval.precision(posClassLabel));
			          System.out.println("Recall " + eval.recall(posClassLabel));
			          System.out.println("F1 " + eval.fMeasure(posClassLabel));
			          System.out.println();
			          System.out.println(eval.toMatrixString());
			          System.out.println();
			          System.out.println();
			        } catch (Exception e) {
			        	System.out.println("There was an exception.");
			             System.err.println(e.getMessage());
			             e.printStackTrace();
			        }
	    	  }
	      }
     
	    }
	  }
 
 //Sarah
 public void testClassifierFamilyTFRBM_Sarah() {
	    setDatasets();
	    TFRBMFactory family = new TFRBMFactory();
	    for (int i = 0; i < family.size(); i++) {
	    	
		      FRMClassifier c = family.buildClassifier(i);
		      System.out.println((i + 1) + " De la familia: " + c.name());
		      
		      for(int d = 0; d < dataNames.length; d++){
			      
		    	  if(d != 5){ // ignore Suramin
			    	  dataName = dataNames[d];
			    	  path = paths[d];
			    	  
			    	  System.out.println("Dataset: " + dataName);
			    	  
				      try {
				          Evaluation eval = null;
				          for (int fold = 1; fold <= folds; fold++) {
				            String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
				            String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
				            Instances trainData = DataSource.read(trainFileName);
				            Instances testData = DataSource.read(testFileName);
				            resetWeights(trainData);
				            resetWeights(testData);
				            trainData.setClassIndex(trainData.numAttributes() - 1);
				            testData.setClassIndex(testData.numAttributes() - 1);
				            c.buildClassifier(trainData);
				            //System.out.println(String.format("Built on fold %d!", fold));
				            if (eval == null)
				              eval = new Evaluation(trainData);
				            else
				              eval.setPriors(trainData);
				            eval.evaluateModel(c, testData);
				            //System.out.println(String.format("Evaluation on fold %d done!", fold));
				          }
				          System.out.println();
				          System.out.println("***** Micro-Average Performance *****");
				          System.out.println();
				          System.out.println("Accuracy " + eval.pctCorrect());
				          System.out.println("Kappa " + eval.kappa());
				          System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
				          System.out.println("Precision " + eval.precision(posClassLabel));
				          System.out.println("Recall " + eval.recall(posClassLabel));
				          System.out.println("F1 " + eval.fMeasure(posClassLabel));
				          System.out.println();
				          System.out.println(eval.toMatrixString());
				          System.out.println();
				          System.out.println();
				        } catch (Exception e) {
				        	System.out.println("There was an exception.");
				             System.err.println(e.getMessage());
				             e.printStackTrace();
				        }
		    	  }
		      }    
	    }
	  }
 
 	public void test_factory_new() { 		
 		setDatasets();
 		
 		double beta = 2;
 		//String[] bagToClassAppMemberships = {"-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"};
 		String[] bagToClassAppMemberships = {"-OWAmax", "-OWAmaxAdd"};
 		//String[] instanceToClassAppMemberships = {"-STDminmax", "-OWAminmax", "-OWAminmaxAdd"};
 		String[] instanceToClassAppMemberships = {"-OWAminmaxAdd"};
 		//String[] instanceToClassMemberships = {"-Ave", "-OWAmax", "-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		String[] instanceToClassMemberships = {"-OWAmaxAdd"};
 		//String[] instanceToBagMemberships = {"-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"};
 		String[] instanceToBagMemberships = {"-Max"};
 		
 		int count = 0;
 		for(int i = 0; i < bagToClassAppMemberships.length; i++){
 			String bagToClassApp = bagToClassAppMemberships[i];
 			for(int j = 0; j < instanceToClassAppMemberships.length; j++){
 				String instanceToClassApp = instanceToClassAppMemberships[j];
 				for(int k = 0; k < instanceToClassMemberships.length; k++){
 					String instanceToClass = instanceToClassMemberships[k];
 					for(int l = 0; l < instanceToBagMemberships.length; l++){
 						String instanceToBag = instanceToBagMemberships[l];
 						
 						TFRIMFactory_Sarah c = new TFRIMFactory_Sarah(bagToClassApp, instanceToClassApp, 
 								instanceToClass, instanceToBag, beta);
 						count++;
 						
 						System.out.println("Classifier " + count + ": TFRIM" + bagToClassApp + instanceToClassApp + 
 																	 instanceToClass + instanceToBag);
 						
 					    for(int d = 0; d < dataNames.length; d++){
 						      
				    	  if(d != 5){ // ignore Suramin
					    	  dataName = dataNames[d];
					    	  path = paths[d];
					    	  
					    	  System.out.println("Dataset: " + dataName); 						
					
						    try {
						      Evaluation eval = null;
						      for (int fold = 1; fold <= folds; fold++) {
						        String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
						        String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
						        Instances trainData = DataSource.read(trainFileName);
						        Instances testData = DataSource.read(testFileName);
						        resetWeights(trainData);
						        resetWeights(testData);
						        trainData.setClassIndex(trainData.numAttributes() - 1);
						        testData.setClassIndex(testData.numAttributes() - 1);
						        c.buildClassifier(trainData);
						        if (eval == null)
						          eval = new Evaluation(trainData);
						        else
						          eval.setPriors(trainData);
						        eval.evaluateModel(c, testData);
						      }
						      System.out.println();
						      System.out.println("***** Micro-Average Performance *****");
						      System.out.println();
						      System.out.println("Accuracy " + eval.pctCorrect());
						      System.out.println("Kappa " + eval.kappa());
						      System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
						      System.out.println("Precision " + eval.precision(posClassLabel));
						      System.out.println("Recall " + eval.recall(posClassLabel));
						      System.out.println("F1 " + eval.fMeasure(posClassLabel));
						      System.out.println();
						      System.out.println(eval.toMatrixString());
						      System.out.println();
						      System.out.println();
						    } catch (Exception e) {
						         System.err.println(e.getMessage());
						    }
					 						
				    	  }
 						}
 					}
 				}
 			}
 		}
	  }
 	
 	
 	public void test_factory_imbalanced() { 		
 		setDatasetsImbalanced();
 		
 		double beta = 1;
 		String[] bagToClassAppMemberships = {"-OWAmax"};
 		String[] instanceToClassAppMemberships = {"-STDminmax"};
 		String[] instanceToClassMemberships = {"-OWAmaxAdd"};
 		String[] instanceToBagMemberships = {"-Max"};
 		
 		int count = 0;
 		for(int i = 0; i < bagToClassAppMemberships.length; i++){
 			String bagToClassApp = bagToClassAppMemberships[i];
 			for(int j = 0; j < instanceToClassAppMemberships.length; j++){
 				String instanceToClassApp = instanceToClassAppMemberships[j];
 				for(int k = 0; k < instanceToClassMemberships.length; k++){
 					String instanceToClass = instanceToClassMemberships[k];
 					for(int l = 0; l < instanceToBagMemberships.length; l++){
 						String instanceToBag = instanceToBagMemberships[l];
 						
 						TFRIMFactory_Sarah c = new TFRIMFactory_Sarah(bagToClassApp, instanceToClassApp, 
 								instanceToClass, instanceToBag, beta);
 						count++;
 						
 						System.out.println("Classifier " + count + ": TFRIM" + bagToClassApp + instanceToClassApp + 
 																	 instanceToClass + instanceToBag);
 						
 					    for(int d = 4; d < dataNames.length; d++){
 						      
					    	  dataName = dataNames[d];
					    	  path = paths[d];
					    	  
					    	  System.out.println("Dataset: " + dataName); 						
					
						    try {
						      Evaluation eval = null;
						      for (int fold = 1; fold <= folds; fold++) {
						        String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
						        String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
						        Instances trainData = DataSource.read(trainFileName);
						        Instances testData = DataSource.read(testFileName);
						        resetWeights(trainData);
						        resetWeights(testData);
						        trainData.setClassIndex(trainData.numAttributes() - 1);
						        testData.setClassIndex(testData.numAttributes() - 1);
						        c.buildClassifier(trainData);
						        System.out.println("Built fold " + fold);
						        if (eval == null)
						          eval = new Evaluation(trainData);
						        else
						          eval.setPriors(trainData);
						        eval.evaluateModel(c, testData);
						        System.out.println("Evaluated fold " + fold);
						      }
						      System.out.println();
						      System.out.println("***** Micro-Average Performance *****");
						      System.out.println();
						      System.out.println("Accuracy " + eval.pctCorrect());
						      System.out.println("Kappa " + eval.kappa());
						      System.out.println("AUC " + eval.areaUnderROC(posClasses[d]));
						      System.out.println("Precision " + eval.precision(posClasses[d]));
						      System.out.println("Recall " + eval.recall(posClasses[d]));
						      System.out.println("F1 " + eval.fMeasure(posClasses[d]));
						      System.out.println("Gmean " + Math.sqrt(eval.truePositiveRate(0) * eval.trueNegativeRate(0)));
						      System.out.println();
						      System.out.println(eval.toMatrixString());
						      System.out.println();
						      System.out.println();
						    } catch (Exception e) {
						         System.err.println(e.getMessage());
						    }
					 						
				    	  }
 					
 					}
 				}
 			}
 		}
	  }
 	
 	
 	public void test_factory_new_bag_based() { 		
 		setDatasets();
 		
 		String[] bagToClassAppMemberships = {"-OWAminmaxAdd"};
 		//String[] bagToClassMemberships = {"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		String[] bagToClassMemberships = {"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		//String[] bagSimilarity = {"-Haus", "-OWAHaus", "-AveHaus", "-AveOWAHaus", "-OWAHausAdd", "-AveOWAHausAdd"};
 		String[] bagSimilarities = {"-Haus", "-OWAHaus", "-AveOWAHaus", "-OWAHausAdd", "-AveOWAHausAdd"};

 		int count = 0;
 		for(int i = 0; i < bagToClassAppMemberships.length; i++){
 			String bagToClassApp = bagToClassAppMemberships[i];
 				for(int k = 0; k < bagToClassMemberships.length; k++){
 					String bagToClass = bagToClassMemberships[k];
 					
 					for(int l = 0; l < bagSimilarities.length; l++){
 						
 						String bagSimilarity = bagSimilarities[l];
 						
						TFRBMFactory_Sarah c = new TFRBMFactory_Sarah(bagToClass, bagToClassApp, bagSimilarity);
						count++;
						
						System.out.println("Classifier " + count + ": TFRBM" + bagToClassApp + bagToClass+bagSimilarity);
						
					    for(int d = 0; d < dataNames.length; d++){
						      
				    	  if(d != 5){ // ignore Suramin
					    	  dataName = dataNames[d];
					    	  path = paths[d];
					    	  
					    	  System.out.println("Dataset: " + dataName); 						
					
						    try {
						      Evaluation eval = null;
						      for (int fold = 1; fold <= folds; fold++) {
						        String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
						        String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
						        Instances trainData = DataSource.read(trainFileName);
						        Instances testData = DataSource.read(testFileName);
						        resetWeights(trainData);
						        resetWeights(testData);
						        trainData.setClassIndex(trainData.numAttributes() - 1);
						        testData.setClassIndex(testData.numAttributes() - 1);
						        c.buildClassifier(trainData);
						        if (eval == null)
						          eval = new Evaluation(trainData);
						        else
						          eval.setPriors(trainData);
						        eval.evaluateModel(c, testData);
						      }
						      System.out.println();
						      System.out.println("***** Micro-Average Performance *****");
						      System.out.println();
						      System.out.println("Accuracy " + eval.pctCorrect());
						      System.out.println("Kappa " + eval.kappa());
						      System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
						      System.out.println("Precision " + eval.precision(posClassLabel));
						      System.out.println("Recall " + eval.recall(posClassLabel));
						      System.out.println("F1 " + eval.fMeasure(posClassLabel));
						      System.out.println();
						      System.out.println(eval.toMatrixString());
						      System.out.println();
						      System.out.println();
						    } catch (Exception e) {
						         System.err.println(e.getMessage());
						    }
					 						
				    	  }
					    }
					}
				}
			}
	  }
  
 	public void test_factory_imbalanced_bag_based() { 		
 		setDatasetsImbalanced();
 		
 		String[] bagToClassAppMemberships = {"-OWAminmax", "-STDminmax","-OWAminmaxAdd"};
 		//String[] bagToClassMemberships = {"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		String[] bagToClassMemberships = {"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		//String[] bagSimilarity = {"-Haus", "-OWAHaus", "-AveHaus", "-AveOWAHaus", "-OWAHausAdd", "-AveOWAHausAdd"};
 		String[] bagSimilarities = {"-Haus", "-OWAHaus", "-AveOWAHaus", "-OWAHausAdd", "-AveOWAHausAdd"};
 		
 		int count = 0;
 		for(int i = 0; i < bagToClassAppMemberships.length; i++){
 			String bagToClassApp = bagToClassAppMemberships[i];
 				for(int k = 0; k < bagToClassMemberships.length; k++){
 					String bagToClass = bagToClassMemberships[k];
 					
					for(int l = 0; l < bagSimilarities.length; l++){
 						
 						String bagSimilarity = bagSimilarities[l];
 						
						TFRBMFactory_Sarah c = new TFRBMFactory_Sarah(bagToClass, bagToClassApp, bagSimilarity);
						count++;
						
						System.out.println("Classifier " + count + ": TFRBM" + bagToClassApp + bagToClass+bagSimilarity);
						
					    for(int d = 4; d < dataNames.length; d++){
						      
					    	  dataName = dataNames[d];
					    	  path = paths[d];
					    	  
					    	  System.out.println("Dataset: " + dataName); 						
					
						    try {
						      Evaluation eval = null;
						      for (int fold = 1; fold <= folds; fold++) {
						        String trainFileName = path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "train");
						        String testFileName =  path + outputDir + String.format("%s-f%d-%s.arff", dataName, fold, "test");
						        Instances trainData = DataSource.read(trainFileName);
						        Instances testData = DataSource.read(testFileName);
						        resetWeights(trainData);
						        resetWeights(testData);
						        trainData.setClassIndex(trainData.numAttributes() - 1);
						        testData.setClassIndex(testData.numAttributes() - 1);
						        c.buildClassifier(trainData);
						        if (eval == null)
						          eval = new Evaluation(trainData);
						        else
						          eval.setPriors(trainData);
						        eval.evaluateModel(c, testData);
						      }
						      System.out.println();
						      System.out.println("***** Micro-Average Performance *****");
						      System.out.println();
						      System.out.println("Accuracy " + eval.pctCorrect());
						      System.out.println("Kappa " + eval.kappa());
						      System.out.println("AUC " + eval.areaUnderROC(posClasses[d]));
						      System.out.println("Precision " + eval.precision(posClasses[d]));
						      System.out.println("Recall " + eval.recall(posClasses[d]));
						      System.out.println("F1 " + eval.fMeasure(posClasses[d]));
						      System.out.println("Gmean " + Math.sqrt(eval.truePositiveRate(0) * eval.trueNegativeRate(0)));
						      System.out.println();
						      System.out.println(eval.toMatrixString());
						      System.out.println();
						      System.out.println();
						    } catch (Exception e) {
						         System.err.println(e.getMessage());
						    }
					    }			
			    	  }
				}
			}
	  }
  
  
  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    Launcher L = new Launcher();
    //L.testClassifier_5CV();
    
    // Sarah
    //System.out.println("-------------- Instance-based fuzzy MIL classifiers --------------");
    //L.testClassifierFamilyTFRIM_Sarah();
    //System.out.println("-------------- Bag-based fuzzy MIL classifiers --------------");
    //L.testClassifierFamilyTFRBM_Sarah(); 
    
    L.test_factory_imbalanced();
  }
}
