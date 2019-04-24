/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.multipleOperators;

import core.Evaluable;
import core.Iterator;
import core.MultipleOperator;
import core.WeighingMaker;
import core.Var;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Class for the Ordered Weighted Operator.
 *
 * @author Danel
 */
public class OWA extends MultipleOperator implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * specify the weight of the operands.
   */
  private WeighingMaker weighingMaker;

  /**
   * index of each operand in the ascending sorting.
   */
   private int[] operandSortedIndex;

  /**
   * Constructor for multiple OWA operator.
   *
   * @param weighingMaker object which specify the weight of each operand.
   * @param iterator the object which will control the iterations.
   * @param expression that gives values to the operands.
   */
  public OWA(WeighingMaker weighingMaker, Iterator iterator, Evaluable expression) {
    super(iterator, expression);
    this.weighingMaker = weighingMaker;
  }

  /**
   * Constructor for multiple OWA operator from an iterator and an expression
   * placeholder.
   *
   * @param weighingMaker object which specify the weight of each operand.
   * @param iterator the object which will control the iterations.
   * @param expression placeholder for the expression that gives values to the
   * operands.
   */
  public OWA(WeighingMaker weighingMaker, Iterator iterator, Var<Evaluable> expression) {
    super(iterator, expression);
    this.weighingMaker = weighingMaker;
  }

  /**
   * Gets the value of the operand in the position "index" as sorted in descending order.
   * @param index integer between 0 and the number of operands minus one.
   * @return the value of the operand.
   */
  private double decreasingOperandValue(int index) {
    //return operandSortedIndex[operandSortedIndex.length - index - 1];
	  return operand(operandSortedIndex[operandSortedIndex.length - index - 1]);
  }

  /**
   * Sort the operands in ascending order.
   */
  private void sortOperands() {
    operandSortedIndex = weka.core.Utils.sort(getOperands());
  }

  /**
   * Evaluate the OWA operator.
   *
   * @return the OWA value of its operands.
   */
  public double evaluate() {
    evaluateOperands();
    double[] weights = weighingMaker.getWeightVector(numOperands());
    sortOperands();
    double sum = 0;
    for (int i = 0; i < numOperands(); i++) {  
      sum += weights[i] * decreasingOperandValue(i);
    }
    return sum;
  }

}
