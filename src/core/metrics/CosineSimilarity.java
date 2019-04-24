/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package core.metrics;

import core.InstanceSimilarity;
import core.VarReader;
import weka.core.DenseInstance;
import weka.core.SparseInstance;
import weka.core.Instance;

/**
 * Class for the cosine similarity between instances.
 *
 * @author Danel
 */
public class CosineSimilarity extends InstanceSimilarity {

  /** for serialization */
  private static final long serialVersionUID = 1L;

  /**
   * Creates a function which calculates the cosine similarity between two instances.
   *
   * @param x variable for the first instance.
   * @param y variable for the second instance.
   */
  public CosineSimilarity(VarReader <Instance> x, VarReader <Instance> y) {
    super(x, y);
  }

  private double denseCos(DenseInstance a, DenseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numAttributes(); i++) {
      ab += a.value(i) * b.value(i);
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  private double sparseCos(SparseInstance a, SparseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numValues(); i++) {
      a2 += a.valueSparse(i) * a.valueSparse(i);
    }
    if (a2 == 0) return 0;
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      b2 += b.valueSparse(i) * b.valueSparse(i);
      double av = a.value(bi);
      if (av != 0)
        ab += av * b.valueSparse(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  private double cos(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // I assume b is also SparseInstance
      return sparseCos((SparseInstance)a, (SparseInstance)b);
    else                                    // I assume both a and b are DenseInstance
      return denseCos((DenseInstance)a, (DenseInstance)b);
  }

  @Override
  public double evaluate() {
    Instance a = (Instance)argument(1).object();
    Instance b = (Instance)argument(2).object();
    return cos(a, b);
  }

}