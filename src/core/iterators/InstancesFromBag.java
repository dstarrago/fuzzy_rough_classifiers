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
 * Class for a dataset of instances corresponding to a MIL bag.
 *
 * @author Danel
 */
public class InstancesFromBag extends Iterator {

  /**
   * The bag containing the instances to iterate.
   */
  private VarReader<Instance> bag;

  /**
   * Creates the iterator.
   *
   * @param item variable that will hold the current instance of the iteration.
   * @param dataset the instances over which iteration will act on.
   */
  public InstancesFromBag(VarWriter<Instance> item, VarReader<Instance> bag) {
    super(item);
    this.bag = bag;
  }

  /**
   * Return the dataset of instances encapsulated.
   *
   * @return an Instances object representing the dataset of instances encapsulated. .
   */
  @Override
  public Instances dataset() {
    return bag.object().relationalValue(1);
  }

}
