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
 * Class for iterating bags from a dataset.
 *
 * @author Danel
 */
public class AllSourceBags extends Iterator {

  /**
   * The dataset of instance to iterate.
   */
  private VarReader<Instances> dataset;

  /**
   * Creates the iterator.
   *
   * @param item variable that will hold the current instance of the iteration.
   * @param dataset the instances over which iteration will act on.
   */
  public AllSourceBags(VarWriter<Instance> item, VarReader<Instances> set) {
    super(item);
    this.dataset = set;
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

}
