/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package experimenter;

/**
 * Represents an entry in a data collection.
 *
 * @author Danel
 */
public class DataEntry {

  /**
   * Mask for the validation type of the data.
   */
  public static String VALIDATION_TYPE = "%dCV";

  /**
   * Name of the data.
   */
  private String name;

  /**
   * Path where the data is located.
   */
  private String path;

  /**
   * Folder where the data is located. The folder is after the path.
   */
  private String folder;

  /**
   * Number of cross validation scheme folds.
   */
  private int numFolds;

  /**
   * Label's index of the positive class.
   */
  private int posClassLabel;

  /**
   * Mask for the validation scheme directory.
   */
  private String validationDirMask;

  /**
   * Mask for the file name.
   */
  private String fileNameMask;

  /**
   * Creates a data entry supplying all the parameters.
   * @param name name of the data.
   * @param path path where the data is located.
   * @param folder folder where the data is located.
   * @param numFolds number of cross validation scheme folds.
   * @param posClassLabel label's index of the positive class.
   * @param validationDirMask mask for the validation scheme directory.
   * @param fileNameMask mask for the file name.
   */
  public DataEntry(String name, String path, String folder, int numFolds,
          int posClassLabel, String validationDirMask, String fileNameMask) {
    this.name = name;
    this.path = path;
    if (!path.endsWith("/")) this.path += "/";
    this.folder = folder;
    if (!folder.endsWith("/")) this.folder += "/";
    this.numFolds = numFolds;
    this.posClassLabel = posClassLabel;
    this.validationDirMask = validationDirMask;
    if (!validationDirMask.endsWith("/")) this.validationDirMask += "/";
    this.fileNameMask = fileNameMask;
  }

  /**
   * Creates a data entry with two default parameters: the mask for the validation
   * scheme directory and the mask for the file name.
   * @param name name of the data.
   * @param path path where the data is located.
   * @param folder folder where the data is located.
   * @param numFolds number of cross validation scheme folds.
   * @param posClassLabel label's index of the positive class.
   */
  public DataEntry(String name, String path, String folder, int numFolds,
          int posClassLabel) {
    this(name, path, folder, numFolds, posClassLabel, "%d-Folds-CV/", "%s-f%d-%s.arff");
  }

  /**
   * Creates a data entry with three default parameters: the mask for the validation
   * scheme directory, the mask for the file name and label's index of the positive class.
   * @param name name of the data.
   * @param path path where the data is located.
   * @param folder folder where the data is located.
   * @param numFolds number of cross validation scheme folds.
   */
  public DataEntry(String name, String path, String folder, int numFolds) {
    this(name, path, folder, numFolds, 1, "%d-Folds-CV/", "%s-f%d-%s.arff");
  }

  /**
   * Access method for the name.
   * @return the name of the data.
   */
  public String name() {
    return name;
  }

  /**
   * Access method for the path.
   * @return the path where the data is located.
   */
  public String path() {
    return path;
  }

  /**
   * Access method for the folder.
   * @return the folder where the data is located.
   */
  public String folder() {
    return folder;
  }

  /**
   * Access method for the number of folds.
   * @return the number of cross validation scheme folds.
   */
  public int numFolds() {
    return numFolds;
  }

  /**
   * Access method for the number positive class label.
   * @return the label's index of the positive class.
   */
  public int posClassLabel() {
    return posClassLabel;
  }

  /**
   * Access method for the validation scheme directory.
   * @return the mask for the validation scheme directory.
   */
  public String validationDirMask() {
    return validationDirMask;
  }

  /**
   * Access method for the file name mask.
   * @return the mask for the file name.
   */
  public String fileNameMask() {
    return fileNameMask;
  }

  /**
   * Returns the file name of the partition given the fold and stage.
   * @param stage can be one of the strings "train" and "test".
   * @param fold is a number representing the fold of the cross validation.
   * @return an string with the full file name.
   */
  public String fold(String stage, int fold) {
    String s = path + folder + String.format(validationDirMask, numFolds) +
            String.format(fileNameMask, name, fold, stage);
    return s;
  }

  /**
   * Returns the file name of the training partition given the fold.
   * @param fold is a number representing the fold of the cross validation.
   * @return an string with the full file name of the training partition.
   */
  public String trainFold(int fold) {
    return fold("train", fold);
  }

  /**
   * Returns the file name of the testing partition given the fold.
   * @param fold is a number representing the fold of the cross validation.
   * @return an string with the full file name of the testing partition.
   */
  public String testFold(int fold) {
    return fold("test", fold);
  }

}
