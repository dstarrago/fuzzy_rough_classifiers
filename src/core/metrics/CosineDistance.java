/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.unaryOperators.Complement;
import core.Evaluable;
import core.InstanceDistance;
import core.VarReader;
import weka.core.Instance;

/**
 * Class for the cosine distance between instances. It is defined as the complement
 * of the cosine similarity.
 *
 * @author Danel
 */
public class CosineDistance extends InstanceDistance {

  /**
   * Mathematical expression defining this function.
   */
  private final Evaluable expression;

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function which calculates the cosine similarity between two instances.
   *
   * @param x variable for the first instance.
   * @param y variable for the second instance.
   */
  public CosineDistance(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
    expression = new Complement( new CosineSimilarity(x, y));
  }

  @Override
  public double evaluate() {
    return expression.evaluate();
  }


}
