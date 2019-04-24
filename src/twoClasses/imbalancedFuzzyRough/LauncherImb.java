package twoClasses.imbalancedFuzzyRough;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LauncherImb {
	
	private final int folds = 5;
	private final String outputDir = String.format("%d-Folds-CV/", folds);
	
	//set up to use all datasets
	 private String[] dataNames;
	 private String[] paths;
	 
	 
	 private void setDatasets(){
		  dataNames = new String[36];
		  paths = new String[36];
		  
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
	      // WIRSel-5
	      dataNames[8] = "WIRSel-5";
	      paths[8] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/30 WIR5/";
	      // WIRSel-6
	      dataNames[9] = "WIRSel-6";
	      paths[9] = "C:/Users/svluyman/Documents/MIL/Data/Imbalanced/31 WIR6/";
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
	 
 	 public void test_factory_new() { 	
 		setDatasets();	

 		double beta = 1;
 		//String[] bagToClassAppMemberships = {"-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"};
 		String[] bagToClassAppMemberships = {"-OWAmax"};
 		//String[] instanceToClassMemberships = {"-Ave", "-OWAmax", "-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		String[] instanceToClassMemberships = {"-OWAmax"};
 		//String[] instanceToBagMemberships = {"-Max", "-OWAmax", "-Ave", "-OWAmaxAdd"};
 		String[] instanceToBagMemberships = {"-Max"};
 		//int[] versions = {1,2,3,4,5,6};
 		int[] versions = {4};
 		double[] gammas = {0.1};
 		
 		int count = 0;
 		for(int i = 0; i < bagToClassAppMemberships.length; i++){
 			String bagToClassApp = bagToClassAppMemberships[i];
			for(int k = 0; k < instanceToClassMemberships.length; k++){
				String instanceToClass = instanceToClassMemberships[k];
				for(int l = 0; l < instanceToBagMemberships.length; l++){
					String instanceToBag = instanceToBagMemberships[l];
					
					for(int m = 0; m < versions.length; m++){					
						int versionIFROWANN = versions[m];
						
						for(int n = 0; n < gammas.length; n++){
							
							double gamma = gammas[n];

							InstanceBased c = new InstanceBased(bagToClassApp,instanceToClass, 
									instanceToBag, beta, versionIFROWANN, gamma);
							count++;
							
							System.out.println("Classifier " + count + ": TFRIM" + bagToClassApp +  
									 instanceToClass + instanceToBag + "-" + versionIFROWANN + "-" + gamma);
												
							for(int d = 30; d < 36; d++){		
							//for(int d = 4; d < dataNames.length; d++){	
					    	  String dataName = dataNames[d];
					    	  String path = paths[d];
					    	  
					    	  System.out.println("Dataset: " + d + ", " + dataName);			    	  
							
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
							      int posClassLabel = c.getPosClass();
							      System.out.println(posClassLabel);
							      System.out.println();
							      System.out.println("***** Micro-Average Performance *****");
							      System.out.println();
							      System.out.println("Accuracy " + eval.pctCorrect());
							      System.out.println("Kappa " + eval.kappa());
							      System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
							      System.out.println("Precision " + eval.precision(posClassLabel));
							      System.out.println("Recall " + eval.recall(posClassLabel));
							      System.out.println("F1 " + eval.fMeasure(posClassLabel));
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

	  }
 	 
 	 
 	public void test_factory_bag_based() { 	
 		setDatasets();	

 		//String[] bagToClassMemberships = {"-Ave", "-OWAmax","-CompAve", "-CompOWAmin", "-OWAmaxAdd", "-CompOWAminAdd"};
 		String[] bagToClassMemberships = {"-OWAmax","-OWAmaxAdd"};
 		//int[] versions = {1,2,3,4,5,6};
 		int[] versions = {1,2,3,6};
 		double[] gammas = {0.1};
 		
 		int count = 0;
 		for(int i = 0; i < bagToClassMemberships.length; i++){
 			String bagToClass = bagToClassMemberships[i];
				
			for(int m = 0; m < versions.length; m++){					
				int versionIFROWANN = versions[m];
				
				for(int n = 0; n < gammas.length; n++){
					
					double gamma = gammas[n];

					BagBased c = new BagBased(bagToClass, versionIFROWANN, gamma);
					count++;
					
					System.out.println("Classifier " + count + ": TFRBM" + bagToClass +  
							 "-" + versionIFROWANN + "-" + gamma);
										
					//for(int d = 4; d < 5; d++){		
					for(int d = 4; d < dataNames.length; d++){	
			    	  String dataName = dataNames[d];
			    	  String path = paths[d];
			    	  
			    	  System.out.println("Dataset: " + d + ", " + dataName);			    	  
					
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
					      int posClassLabel = c.getPosClass();
					      System.out.println(posClassLabel);
					      System.out.println();
					      System.out.println("***** Micro-Average Performance *****");
					      System.out.println();
					      System.out.println("Accuracy " + eval.pctCorrect());
					      System.out.println("Kappa " + eval.kappa());
					      System.out.println("AUC " + eval.areaUnderROC(posClassLabel));
					      System.out.println("Precision " + eval.precision(posClassLabel));
					      System.out.println("Recall " + eval.recall(posClassLabel));
					      System.out.println("F1 " + eval.fMeasure(posClassLabel));
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
	    LauncherImb L = new LauncherImb();	    
	    L.test_factory_new();
	    //L.test_factory_bag_based();
	  }

}
