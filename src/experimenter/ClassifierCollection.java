/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package experimenter;

import java.util.ArrayList;

/**
 * Class to handle a collection of classifiers. Just add to the list as many
 * classifier entries as you want. Then you can access one entry by the method
 * <tt>classifierEntry(int index)</tt>.
 *
 * @author Danel
 */
public class ClassifierCollection {

  /**
   * List of classifier entries.
   */
  private ArrayList<ClassifierEntry> classifierEntries;

  /**
   * Creates the collection.
   */
  public ClassifierCollection() {
    classifierEntries = new ArrayList<ClassifierEntry>();
    compileCollection2();
  }

  /**
   * Almost a full collection of weka MIL classifiers
   */
  private void compileCollection1() {
    /**
     * KNN
     */
    //classifierEntries.add(new ClassifierEntry("CitationKNN (R1,C1)", "weka.classifiers.mi.CitationKNN -R 1 -C 1 -H 1"));
    //classifierEntries.add(new ClassifierEntry("CitationKNN (R3,C3)", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));
	  classifierEntries.add(new ClassifierEntry("CitationKNN (R2,C4)", "weka.classifiers.mi.CitationKNN -R 2 -C 4 -H 1"));
    /**
     * Trees
     */
    //classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));

    /**
     * Rules
     */
    //classifierEntries.add(new ClassifierEntry("MIRI", "weka.classifiers.mi.MIRI -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    
    /**
     * Diverse density
     */
    classifierEntries.add(new ClassifierEntry("MDD", "weka.classifiers.mi.MDD -N 0"));
    classifierEntries.add(new ClassifierEntry("MIDD", "weka.classifiers.mi.MIDD -N 0"));
    classifierEntries.add(new ClassifierEntry("QuickDDIterative", "weka.classifiers.mi.QuickDDIterative -N 0 -S 1.0 -M 1.0 -I 2"));
    
    /**
     * Simple wrappers
     */
    classifierEntries.add(new ClassifierEntry("SimpleMI (A, C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("SimpleMI (G, C4.5)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (A, C4.5)", "weka.classifiers.mi.MIWrapper -P 1 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (G, C4.5)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (M, C4.5)", "weka.classifiers.mi.MIWrapper -P 3 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    
    /**
     * Wrapper with attribute space mapping
     */
    classifierEntries.add(new ClassifierEntry("MILES (C4.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MILES (Adaboost)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10"));
    
    /**
     * Boosting
     */
    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MITI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MIRI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MIRI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    
    /**
     * Logistic regression
     */
    classifierEntries.add(new ClassifierEntry("MILR (A)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 1"));
    classifierEntries.add(new ClassifierEntry("MILR (G)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 2"));
    classifierEntries.add(new ClassifierEntry("MILR (S)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 0"));

    /**
     * Support vector machines
     */
    classifierEntries.add(new ClassifierEntry("MISVM (K1)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISVM (K2)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K1)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K2)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
  }

  /**
   * Dedicated compilation for Suramin dataset.
   * Almost a full collection of weka MIL classifiers but
   * MITI based classifiers have special settings for Suramin dataset.
   */
  private void compileCollection2() {
    /**
     * KNN
     */
    classifierEntries.add(new ClassifierEntry("CitationKNN (R1,C1)", "weka.classifiers.mi.CitationKNN -R 1 -C 1 -H 1"));
    classifierEntries.add(new ClassifierEntry("CitationKNN (R3,C3)", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));

    /**
     * Trees
     */
    classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Rules
     */
    classifierEntries.add(new ClassifierEntry("MIRI", "weka.classifiers.mi.MIRI -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Diverse density
     */
    classifierEntries.add(new ClassifierEntry("MINND", "weka.classifiers.mi.MINND -K 1 -S 1 -E 1"));
    classifierEntries.add(new ClassifierEntry("MDD", "weka.classifiers.mi.MDD -N 0"));
    classifierEntries.add(new ClassifierEntry("MIDD", "weka.classifiers.mi.MIDD -N 0"));
    classifierEntries.add(new ClassifierEntry("QuickDDIterative", "weka.classifiers.mi.QuickDDIterative -N 0 -S 1.0 -M 1.0 -I 2"));

    /**
     * Simple wrappers
     */
    classifierEntries.add(new ClassifierEntry("SimpleMI (A, C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("SimpleMI (G, C4.5)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (A, C4.5)", "weka.classifiers.mi.MIWrapper -P 1 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (G, C4.5)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (M, C4.5)", "weka.classifiers.mi.MIWrapper -P 3 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));

    /**
     * Wrapper with attribute space mapping
     */
    classifierEntries.add(new ClassifierEntry("MILES (C4.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));

    /**
     * Boosting
     */
    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MITI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MIRI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MIRI -- -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Logistic regression
     */
    classifierEntries.add(new ClassifierEntry("MILR (A)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 1"));
    classifierEntries.add(new ClassifierEntry("MILR (G)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 2"));
    classifierEntries.add(new ClassifierEntry("MILR (S)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 0"));

    /**
     * Support vector machines
     */
    classifierEntries.add(new ClassifierEntry("MISVM (K1)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISVM (K2)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K1)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K2)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
  }

  /**
   * Number of classifiers in this collection.
   * @return the number of classifiers in this collection.
   */
  public int numClassifiers() {
    return classifierEntries.size();
  }

  /**
   * Return a given classifier entry.
   * @param index the index of the classifier entry in the list.
   * @return the classifier entry given by the index.
   */
  public ClassifierEntry classifierEntry(int index) {
    return classifierEntries.get(index);
  }

}
