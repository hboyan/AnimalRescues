'''
Identify the problem: What features make shelter animals least likely to get adopted
Acquire the data: Austin animal shelters
Parse the data (EDA)
Mine the data (sample, format, clean, slice, combine, new columns)
Refine (trends, outliers, descriptive stats, inferential stats, document/transform data)
Build model (select, build, evaluate, refine)
Present results
'''

'''
Presentation:
Why is this important?
How did I figure this out (approach, story, process)
What's the point? What's next?

PREM (Point, Reason, Explain, Point)
'''

'''
Model selection:
1. Always have a shortlist of models (5-10)
2. Look at model differences list


Models are different:
1. Underlying data (where does it come from?)
2. Model assumptions
3. Memory intensiveness/model speed (involves frequency of retraining)
4. Interpretability (ease of understanding)
5. Granular outputs

Specific problems and solutions:
1. Unbalanced classes (class weights, logistic regression, SVM, NB, KNN, DT, RF)
    - if binary, use ROC/AUC
    - penalized models (penalized SVM/LDA)
2. Data Size
    - Small dataset (<2k samples): high bias/low variance models (opposite would overfit, too small of a representation,
        means/stds of small test sets would be wildly different from whole set in cross val)
        - parametric/non-stochastic models (otherwise tend to underfit): linear regression, logistic regression, SVM, NB, KNN, GLM
    - Large dataset: low bias/high variance models (to avoid underfitting)
        - non-parametric models (otherwise tend to overfit): Neural Nets, DT, RF, boosted

3. Interpretability
    - DT, KNN, RF (70%), linear regression, logistic regression (60%), NB, Markhov, K-means (80%), LDA

4. Mix of orders of data (nominal, ordinal, ratio, )
    - dummy variables/encode
    - scale
    - like dummy variables: DT, RF, NB
    - bad at dummy variables: KNN, PCA, K-means, SVM... (spatial relations) (unless you can swap out metric with l1/order1 minkowski)

5. Discriminative (decision boundaries) vs. Generative (how x and y were generated)
    - D: classification, SVM, neural nets, DT/RF, most models
    - G: regression, bayesian

'''

'''
Bootcamp buzzwords:
- RECENT/CURRENT
- fast-paced learning (project based/lecture style)
- capstone project (real world data, communication skills, teams)
- specific skills (learning algorithms, data mining/science, statistical modeling, analytical skills)
- python and SQL (PostGRES)
- big data algorithms 
'''
