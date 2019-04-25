# Fuzzy-Rough Classifiers
Framework for developing fuzzy and fuzzy-rough algorithms for multiple-instance classification problems

The framework facilitates the design of multiple-instance classification algorithms that are based on fuzzy-sets and rough-sets. In contrasts with regular classification problems, in which each example has a unique description, in multiple-instance classification (MIC) problems, each example has many descriptions. In MIC, each example is called <em>bag</em>, and each description of a bag is an <em>instance</em>. In fuzzy multi-instance classifiers, each class is regarded as a fuzzy set to which every bag has a degree of membership. When classifying an unseen bag, the classifiers calculate its membership degree to each class and assign it to the class for which this value is largest. In the framework we define java classes for each mathematical concept involved in the calculus of class memberships:

- similarity and distance functions between instances
- similarity and distance functions between bags
- numeric operators for two or more operands (ex., addition, multiplication, maximun, average)
- fuzzy operators (implicators, TNorms)
- aggregators that iterate over, for example, all instances in a bag, or a set of bags, applying a numerical operation on them
- weighting schemes for weighted average aggregation
- bag membership and class membership functions


