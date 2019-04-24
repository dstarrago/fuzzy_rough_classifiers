/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package oneClass.fuzzyRoughClassifiers;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

/**
 * Class for testing the classifiers.
 *
 * @author Danel
 */
public class Launcher {

  private String dataName = "Musk1";
  private String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/"
          + "_Colección MIL #1/01 Musk1/";
  private final int folds = 5;
  private final String outputDir = String.format("%d-Folds-CV/", folds);
  private final int posClassLabel = 1;
  
  
//set up to use all datasets
 private String[] dataNames;
 private String[] paths;
 
 
 private void setDatasets(){
	  dataNames = new String[42];
	  paths = new String[42];
	  
	  // Musk1
	  dataNames[0] = "Musk1";
     paths[0] = "C:/Users/svluyman/Documents/MIL/Data/01 Musk1/";
	  // Musk2
	  dataNames[1] = "Musk2";
     paths[1] = "C:/Users/svluyman/Documents/MIL/Data/02 Musk2/";	  
	  // Atoms
	  dataNames[2] = "Atoms";
     paths[2] = "C:/Users/svluyman/Documents/MIL/Data/03 Atoms/";	  
	  // Bonds
	  dataNames[3] = "Bonds";
     paths[3] = "C:/Users/svluyman/Documents/MIL/Data/04 Bonds/";	  
	  // Chains
	  dataNames[4] = "Chains";
     paths[4] = "C:/Users/svluyman/Documents/MIL/Data/05 Chains/";	  
//	  // Suramin
//	  dataNames[5] = "Suramin";
//     paths[5] = "C:/Users/svluyman/Documents/MIL/Data/06 Suramin/";	  
	  // Elephant
	  dataNames[6] = "Elephant";
     paths[6] = "C:/Users/svluyman/Documents/MIL/Data/08 Elephant/";	  
	  // Fox
	  dataNames[7] = "Fox";
     paths[7] = "C:/Users/svluyman/Documents/MIL/Data/09 Fox/";	  
	  // Tiger
	  dataNames[8] = "Tiger";
     paths[8] = "C:/Users/svluyman/Documents/MIL/Data/10 Tiger/";	  
	  // EastWest
	  dataNames[9] = "EastWest";
     paths[9] = "C:/Users/svluyman/Documents/MIL/Data/11 EastWest/";	  
	  // WestEast
	  dataNames[10] = "WestEast";
     paths[10] = "C:/Users/svluyman/Documents/MIL/Data/12 WestEast/";	  
	  // AntDrugs5
	  dataNames[11] = "AntDrugs5";
     paths[11] = "C:/Users/svluyman/Documents/MIL/Data/16 AntDrugs5/";	  
	  // AntDrugs10
	  dataNames[12] = "AntDrugs10";
     paths[12] = "C:/Users/svluyman/Documents/MIL/Data/17 AntDrugs10/";	  
	  // AntDrugs20
	  dataNames[13] = "AntDrugs20";
     paths[13] = "C:/Users/svluyman/Documents/MIL/Data/18 AntDrugs20/";	  
	  // TREC9-1 
	  dataNames[14] = "TREC9Sel-1";
     paths[14] = "C:/Users/svluyman/Documents/MIL/Data/19 TREC9-1/";	  
	  // TREC9-2
	  dataNames[15] = "TREC9Sel-2";
     paths[15] = "C:/Users/svluyman/Documents/MIL/Data/20 TREC9-2/";	  
	  // TREC9-3
	  dataNames[16] = "TREC9Sel-3";
     paths[16] = "C:/Users/svluyman/Documents/MIL/Data/21 TREC9-3/";	  
	  // TREC9-4
	  dataNames[17] = "TREC9Sel-4";
     paths[17] = "C:/Users/svluyman/Documents/MIL/Data/22 TREC9-4/";	  
	  // TREC9-7
	  dataNames[18] = "TREC9Sel-7";
     paths[18] = "C:/Users/svluyman/Documents/MIL/Data/23 TREC9-7/";	  
	  // TREC9-9
	  dataNames[19] = "TREC9Sel-9";
     paths[19] = "C:/Users/svluyman/Documents/MIL/Data/24 TREC9-9/";	  
	  // TREC9-10
	  dataNames[20] = "TREC9Sel-10";
     paths[20] = "C:/Users/svluyman/Documents/MIL/Data/25 TREC9-10/";	  
	  // WIR7
	  dataNames[21] = "WIRSel-7";
     paths[21] = "C:/Users/svluyman/Documents/MIL/Data/32 WIR7/";	  
	  // WIR8
	  dataNames[22] = "WIRSel-8";
     paths[22] = "C:/Users/svluyman/Documents/MIL/Data/33 WIR8/";	  
	  // WIR9
	  dataNames[23] = "WIRSel-9";
     paths[23] = "C:/Users/svluyman/Documents/MIL/Data/34 WIR9/";	  
	  // CLJ-16.30.2
	  dataNames[24] = "CLJ-16.30.2";
     paths[24] = "C:/Users/svluyman/Documents/MIL/Data/55 CLJ-16.30.2/";	  
	  // CLJ-16-50-2
	  dataNames[25] = "CLJ-16-50-2";
     paths[25] = "C:/Users/svluyman/Documents/MIL/Data/56 CLJ-16-50-2/";	  
	  // CLJ-80.166.1
	  dataNames[26] = "CLJ-80.166.1";
     paths[26] = "C:/Users/svluyman/Documents/MIL/Data/57 CLJ-80.166.1/";	  
	  // CLJ-80.166.1-Strong
	  dataNames[27] = "CLJ-80.166.1-Strong";
     paths[27] = "C:/Users/svluyman/Documents/MIL/Data/58 CLJ-80.166.1-Strong/";	  
	  // CLJ-80-206-1
	  dataNames[28] = "CLJ-80-206-1";
     paths[28] = "C:/Users/svluyman/Documents/MIL/Data/59 CLJ-80-206-1/";	  
	  // CLJ-160.166.1
	  dataNames[29] = "CLJ-160.166.1";
     paths[29] = "C:/Users/svluyman/Documents/MIL/Data/60 CLJ-160.166.1/";	  
	  // CLJ-160.166.1-Strong
	  dataNames[30] = "CLJ-160.166.1-Strong";
     paths[30] = "C:/Users/svluyman/Documents/MIL/Data/61 CLJ-160.166.1-Strong/";	  
	  // CLJ-160-566-1
	  dataNames[31] = "CLJ-160-566-1";
     paths[31] = "C:/Users/svluyman/Documents/MIL/Data/62 CLJ-160-566-1/";	  
	  // Corel01vs02
	  dataNames[32] = "Corel01vs02";
     paths[32] = "C:/Users/svluyman/Documents/MIL/Data/63 Corel01vs02/";	  
	  // Corel01vs03
	  dataNames[33] = "Corel01vs03";
     paths[33] = "C:/Users/svluyman/Documents/MIL/Data/64 Corel01vs03/";	  
	  // Corel01vs04
	  dataNames[34] = "Corel01vs04";
     paths[34] = "C:/Users/svluyman/Documents/MIL/Data/65 Corel01vs04/";	  
	  // Corel01vs05
	  dataNames[35] = "Corel01vs05";
     paths[35] = "C:/Users/svluyman/Documents/MIL/Data/66 Corel01vs05/";	  
	  // Corel02vs03
	  dataNames[36] = "Corel02vs03";
     paths[36] = "C:/Users/svluyman/Documents/MIL/Data/67 Corel02vs03/";	  
	  // Corel02vs04
	  dataNames[37] = "Corel02vs04";
     paths[37] = "C:/Users/svluyman/Documents/MIL/Data/68 Corel02vs04/";	  
	  // Corel02vs05
	  dataNames[38] = "Corel02vs05";
     paths[38] = "C:/Users/svluyman/Documents/MIL/Data/69 Corel02vs05/";	  
	  // Corel03vs04
	  dataNames[39] = "Corel03vs04";
     paths[39] = "C:/Users/svluyman/Documents/MIL/Data/70 Corel03vs04/";	  
	  // Corel03vs05
	  dataNames[40] = "Corel03vs05";
     paths[40] = "C:/Users/svluyman/Documents/MIL/Data/71 Corel03vs05/";	  
     // Corel04vs05
     dataNames[41] = "Corel04vs05";
     paths[41] = "C:/Users/svluyman/Documents/MIL/Data/72 Corel04vs05/";	
 }

  private void resetWeights(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(1);
    }
  }

  public void testClassifier_5CV() {
    OFRBM1 c = new OFRBM1();
    System.out.println("Data " + dataName);
    System.out.println("Classifier " + c.getClass().getName());
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
         System.err.println(e.getMessage());
    }
  }

  
//Sarah
 public void testClassifierFamilyOFRIM_Sarah() {
	    setDatasets();
	    OFRIMFactory family = new OFRIMFactory();
	    for (int i = 0; i < family.size(); i++) {
	      EmptyOFRMClassifier c = (EmptyOFRMClassifier) family.buildClassifier(i);
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
 
 //Sarah
 public void testClassifierFamilyOFRBM_Sarah() {
	    setDatasets();
	    OFRBMFactory family = new OFRBMFactory();
	    for (int i = 0; i < family.size(); i++) {
	    	
		      EmptyOFRMClassifier c = (EmptyOFRMClassifier) family.buildClassifier(i);
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
  
  
  
  
  
  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    Launcher L = new Launcher();
    //L.testClassifier_5CV();
    
   
    // Sarah
   System.out.println("-------------- Bag-based fuzzy MIL classifiers --------------");
   L.testClassifierFamilyOFRBM_Sarah(); 
   System.out.println("-------------- Instance-based fuzzy MIL classifiers --------------");
   L.testClassifierFamilyOFRIM_Sarah();
   
  }
}
