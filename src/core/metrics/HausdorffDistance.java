/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.BagDistance;
import core.Evaluable;
import core.InstanceDistance;
import core.iterators.InstancesFromBag;
import core.Var;
import core.VarReader;
import core.multipleOperators.Max;
import core.multipleOperators.Min;
import core.binaryOperators.Max2;
import java.lang.reflect.Constructor;
import weka.core.Instance;

/**
 * Class for calculating the Hausdorff distance between two bags. The computation
 * finally relies in an instance distance function.
 *
 * @author Danel
 */
public class HausdorffDistance extends BagDistance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Mathematical expression defining this function.
   */
  private Evaluable expression;

  /**
   * Creates a function which calculates the Hausdorff distance between two
   * bags.
   *
   * @param instanceDistanceClass class of the instance distance used.
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public HausdorffDistance(Class<? extends InstanceDistance> instanceDistanceClass,
          VarReader <Instance> X, VarReader <Instance> Y) {
    super(X, Y);
    Var<Instance> x = new Var();
    Var<Instance> y = new Var();
    try {
      /**
       * Instance Distance Constructor
       */
      Constructor<? extends InstanceDistance> cons =
              instanceDistanceClass.getConstructor(VarReader.class, VarReader.class);
      InstanceDistance instanceDistance = cons.newInstance(x, y);
      expression = new Max2(
              new Max(new InstancesFromBag(x, X), new Min(new InstancesFromBag(y, Y), instanceDistance)),
              new Max(new InstancesFromBag(y, Y), new Min(new InstancesFromBag(x, X), instanceDistance)));
    } catch (Exception e) { 
      System.out.println("Problema al construir HausdorffDistance: " + e.toString());
    }
  }

  /**
   * Creates a function which calculates the Hausdorff distance between
   * two bags using the cosine distance between instances.
   *
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public HausdorffDistance(VarReader <Instance> X, VarReader <Instance> Y) {
    this(CosineDistance.class, X, Y);
  }

  /**
   * Evaluates the expression that defines the Hausdorff distance in the
   * current values of the arguments.
   *
   * @return the value of the Hausdorff distance.
   */
  @Override
  public double evaluate() {
    return expression.evaluate();
  }

}
