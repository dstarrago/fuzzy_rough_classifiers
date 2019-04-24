package oneClass.fuzzyClassifiers;

import java.io.Serializable;

public class EmptyOFMClassifier extends OFMClassifier implements Serializable {
	
	  /**
	   * The name of this classifier;
	   */
	  private String name;
	
	  /** for serialization */
	  private static final long serialVersionUID = 1L;

	  public EmptyOFMClassifier() {

	  }
	  
	  /**
	   * Gets the name of this classifier.
	   */
	  public String name() {
	    return name;
	  }

	  /**
	   * Sets the name of this classifier.
	   * 
	   * @param name string that has to identify this classifier.
	   */
	  public void setName(String name) {
	    this.name = name;
	  }

}
