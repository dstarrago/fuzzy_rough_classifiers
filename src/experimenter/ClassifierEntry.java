/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package experimenter;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Utils;

/**
 * Represents an entry in a classifier collection. Use <tt>instantiate()</tt> to
 * create the classifier this entry represents.
 *
 * @author Danel
 */
public class ClassifierEntry {

  /**
   * Name to be shown of the classifier.
   */
  private String name;

  /**
   * Command used to instantiate the classifier. It includes the full class name
   * of the classifier and the list of parameters it uses.
   */
  private String command;

  /**
   * Creator for the classifier entry.
   * @param name the name to be shown of the classifier.
   * @param command command used to instantiate the classifier.
   */
  public ClassifierEntry(String name, String command) {
    this.name = name;
    this.command = command;
  }

  /**
   * Name to be shown of the classifier.
   * @return the name of the classifier.
   */
  public String name() {
    return name;
  }

  /**
   * Creates the classifier this entry represents.
   * @return the classifier.
   * @throws Exception if the classifier construction fail.
   */
  public Classifier instantiate() throws Exception {
    Classifier scheme = null;
    try {
      String[] cmd = Utils.splitOptions(command);
      String schemeFullName = cmd[0];
      cmd[0] = "";
      scheme = AbstractClassifier.forName(schemeFullName, cmd);
    } catch (Exception e) {
      System.err.println("Classifier construction failed: " + e.getMessage());
    }
    return scheme;
  }

}
