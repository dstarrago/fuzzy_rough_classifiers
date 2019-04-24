/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.iterators;

import core.Iterator;
import core.VarReader;
import core.VarWriter;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for a dataset of bags with a given class label in a dataset.
 *
 * @author Danel
 */
public class BagsFromClass extends Iterator {

  /**
   * The dataset of instance to iterate.
   */
  private VarReader<Instances> dataset;

  /**
   * Index of the class label
   */
  private VarReader<Integer> targetClassLabel;

  /**
   * Creates the iterator.
   *
   * @param item variable that will hold the current instance of the iteration.
   * @param dataset the instances over which iteration will act on.
   * @param targetClassLabel index of the class label.
   */
  public BagsFromClass(VarWriter<Instance> item, VarReader<Instances> set,
          VarReader<Integer> targetClassLabel) {
    super(item);
    this.dataset = set;
    this.targetClassLabel = targetClassLabel;
  }

  /**
   * Impose the restriction: if the class label of the current instance is
   * different from the target class label then it can't be the next instance.
   *
   * @return <tt>true</tt> if the current instance do not fulfill the restriction.
   */
  @Override
  public boolean isRestricted() {
    return currentInstance().classValue() != targetClassLabel.object().intValue();
  }

  /**
   * Return the numItems of the dataset.
   */
  @Override
  public int calculateNumItems() {
    int size = 0;
     for (int i = 0; i < dataset().numInstances(); i++) {
       if (dataset().instance(i).classValue() == targetClassLabel.object().intValue())
         size++;
     }
    return size;
  }

  /**
   * Return the dataset of instances encapsulated.
   *
   * @return an Instances object representing the dataset of instances encapsulated.
   */
  @Override
  public Instances dataset() {
    return dataset.object();
  }

  /**
   * Gets the index of the class label.
   *
   * @return the variable holding the index of the class label.
   */
  public VarReader<Integer> targetClassLabel() {
    return targetClassLabel;
  }

}
