/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package experimenter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;

/**
 * Class for testing the classifiers.
 *
 * @author Danel
 */
public class Launcher {

  private DataCollection dataCollection = new DataCollection();
  //private DataCollection dataCollection = new DataCollection("imbalanced");
  private ClassifierCollection classifierCollection = new ClassifierCollection();
  private final int folds = 5;
  //private final int posClassLabel = 1;
  private String outputDir = "C:/Users/svluyman/Documents/MIL/Experiments/";


  private void resetWeights(Instances data) {
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(1);
    }
  }

  public void testClassifier_5CV() {
    ResultsCompiler results = new ResultsCompiler("Experiment_5",
            outputDir, dataCollection, classifierCollection);
    System.out.println(results.name());
    System.out.println();
    for (int k = 0; k < classifierCollection.numClassifiers(); k++) {
    	
    if(k == 10) {

      ClassifierEntry classifierEntry = classifierCollection.classifierEntry(k);
      for (int i = 0; i < dataCollection.numDatasets(); i++) {
      //for (int i = 0; i < 1; i++) {
        try {
          Classifier classifier = classifierEntry.instantiate();
          System.out.println("Classifier " + classifierEntry.name());
          DataEntry dataEntry = dataCollection.dataEntry(i);
          System.out.println("Data " + dataEntry.name());
          System.out.println();
          long trainTimeInNanoSecs = 0;
          long testTimeInNanoSecs = 0;
          Evaluation eval = null;
          for (int fold = 1; fold <= folds; fold++) {
            String trainFileName = dataEntry.trainFold(fold);
            String testFileName =  dataEntry.testFold(fold);
            System.out.println(trainFileName);
            Instances trainData = DataSource.read(trainFileName);
            Instances testData = DataSource.read(testFileName);
            resetWeights(trainData);
            resetWeights(testData);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);

            long startTime = System.nanoTime();
            classifier.buildClassifier(trainData);
            long estimatedTime = System.nanoTime() - startTime;
            trainTimeInNanoSecs += estimatedTime;

            System.out.println(String.format("Built on fold %d!", fold));
            if (eval == null)
              eval = new Evaluation(trainData);
            else
              eval.setPriors(trainData);

            startTime = System.nanoTime();
            eval.evaluateModel(classifier, testData);
            estimatedTime = System.nanoTime() - startTime;
            testTimeInNanoSecs += estimatedTime;

            System.out.println(String.format("Evaluation on fold %d done!", fold));
          }

          double trainTime = trainTimeInNanoSecs / 1.0E9;       //  Train time in seconds
          double testTime = testTimeInNanoSecs / 1.0E9;         //  Test time in seconds

          results.addResult(classifierEntry, dataEntry, eval, trainTime, testTime);

          int posClassLabel = dataEntry.posClassLabel();
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
          System.out.println("Train time " + trainTime);
          System.out.println("Test time " + testTime);
          System.out.println();
	      System.out.println(eval.toMatrixString());
	      System.out.println();          
        } catch (Exception e) {
             System.err.println(e.getMessage());
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
    L.testClassifier_5CV();
  }
}
