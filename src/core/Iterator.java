/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core;

import weka.core.Instances;
import weka.core.Instance;
import java.io.Serializable;

/**
 * Abstract class implementing basic functionality of an iterator over a
 * dataset of instances.
 *
 * @author Danel
 */
public abstract class Iterator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * The current instance in the iteration.
   */
  private VarWriter<Instance> item;

  /**
   * Record the next instance to be delivery
   */
  private int current;

  /**
   * Number of delivered instances
   */
  private int count;

  /**
   * Store the numItems of the dataset
   * Store -1 if the value need to be calculated
   */
  private int numItems = -1;

  /**
   * Creates the iterator.
   *
   * @param item variable that will hold the current instance of the iteration.
   * @param dataset the instances over which iteration will act on.
   */
  public Iterator(VarWriter<Instance> item) {
    this.item = item;
  }

  /**
   * Returns the number of elements in this iterator.
   *
   * @return the number of elements in this iterator, -1 if this number
   * can not be calculated.
   */
  public final int numItems() {
    return numItems;
  }

  /**
   * Return the dataset of instances encapsulated. This method have to be
   * implemented by all descendant classes.
   * @return an Instances object representing the dataset of instances encapsulated.
   */
  abstract public Instances dataset();

  /**
   * Return the current instance in the dataset. Check this instance in order
   * to verify it fulfills the restrictions.
   *
   * @return the current instance.
   */
  protected Instance currentInstance() {
    return dataset().instance(current);
  }

  /**
   * Initialize this iterator. 
   */
  public void initialize() {
    current = 0;
    count = 0;
    numItems = calculateNumItems();
  }

  /**
   * The current element in the iteration.
   *
   * @return the current element in the iteration.
   */
  public Instance item() {
    return item.object();
  }

  /**
   * Gets the read only interface of the variable holding the current instance.
   * @return the instance variable.
   */
  public VarReader<Instance> exemplar() {
    return item;
  }

  /**
   *  Sets the current element in the iteration.
   * @param instance the current element in the iteration.
   */
  public void setItem(Instance instance) {
    item.setObject(instance);
  }

  /**
   * Returns <tt>true</tt> if the iteration has more elements. (In other
   * words, returns <tt>true</tt> if <tt>next</tt> would return an element
   * rather than throwing an exception.)
   *
   * @return <tt>true</tt> if the iterator has more elements.
   */
  public boolean hasNext() {
    return count < numItems;
  }

  /**
   * Returns the next element in the iteration. The implementation of this method
   * should call to <tt>setItem</tt> in order to update the value of the variable,
   * and then return this value.
   *
   * @return the next element in the iteration.
   * @exception NoSuchElementException iteration has no more elements.
   */
  public Instance next() {
    while (isRestricted()) {
      current++;
    }
    count++;
    Instance currentInstance = dataset().instance(current++);
    setItem(currentInstance);
    return currentInstance;
  }

  /**
   * Implements restrictions for this iterator, checking whether the current item
   * can be output in the next iteration. Any extending class that do not output
   * all the items from the dataset should override this method in order to select
   * the items that will be output.
   *
   * @return <tt>true</tt> if the current item is restricted and should no be
   * output, <tt>false</tt> in other case.
   */
  public boolean isRestricted() {
    return false;
  }

  /**
   * Compute the number of elements in this iterator. Any extending class that
   * do not output all the items from the dataset should override this method.
   *
   * @return the number of elements in this iterator, -1 if this number
   * can not be calculated.
   */
  protected int calculateNumItems() {
    return dataset().numInstances();
  }

}
