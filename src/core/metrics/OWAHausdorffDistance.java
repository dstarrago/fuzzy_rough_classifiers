/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.BagDistance;
import core.Evaluable;
import core.InstanceDistance;
import core.iterators.InstancesFromBag;
import core.owaWeighing.LinealMaxWeighing;
import core.owaWeighing.LinealMinWeighing;
import core.MaxWeighingMaker;
import core.MinWeighingMaker;
import core.Var;
import core.VarReader;
import core.multipleOperators.OWA;
import core.binaryOperators.Max2;
import java.lang.reflect.Constructor;
import weka.core.Instance;

/**
 * Class for calculating the Hausdorff distance between two bags using OWA max
 * and min operators. The computation finally relies in an instance distance
 * function. 
 *
 * @author Danel
 */
public class OWAHausdorffDistance extends BagDistance {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Mathematical expression defining this function.
   */
  private Evaluable expression;

  /**
   * Creates a function which calculates the OWA Hausdorff distance between two
   * bags.
   *
   * @param maxWeighing weighing version of the max OWA operator.
   * @param minWeighing weighing version of the min OWA operator.
   * @param instanceDistanceClass class of the instance distance used.
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public OWAHausdorffDistance(MaxWeighingMaker maxWeighing, MinWeighingMaker minWeighing,
          Class<? extends InstanceDistance> instanceDistanceClass,
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
              new OWA(maxWeighing, new InstancesFromBag(x, X),
                  new OWA(minWeighing, new InstancesFromBag(y, Y), instanceDistance)),
              new OWA(maxWeighing, new InstancesFromBag(y, Y),
                  new OWA(minWeighing, new InstancesFromBag(x, X), instanceDistance)));
    } catch (Exception e) { }
  }

  /**
   * Creates a function which calculates the OWA Hausdorff distance between
   * two bags using the cosine distance between instances and the lineal weighing
   * versions of the max and min OWA operators.
   *
   * @param X variable for the first bag.
   * @param Y variable for the second bag.
   */
  public OWAHausdorffDistance(VarReader <Instance> X, VarReader <Instance> Y) {
    this(new LinealMaxWeighing(), new LinealMinWeighing(), CosineDistance.class, X, Y);
  }

  /**
   * Evaluates the expression that defines the OWA Hausdorff distance in the
   * current values of the arguments.
   *
   * @return the value of the OWA Hausdorff distance.
   */
  @Override
  public double evaluate() {
    return expression.evaluate();
  }

}
