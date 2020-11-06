Principal Component Analysis using Eigen Decomposition vs Singular Value Decomposition in Python
------------------------------------------------------------------------------------------------

File PCA.py contains my implementation of the PCA algorithm. To run the
program, execute it as a python script followed by the .csv file path
containing the dataset you'd like to create a tree for, i.e. \$ python
PCA.py

**Program limitations:**

-   Requires Pandas version 0.23.4 or greater. Older versions may work,
     but this was the version I used when programming the algorithm.

<!-- -->

-   The program is built specifically for the supplied cars.mat dataset,
     and as such will only work for that file. This is because manual
     data cleaning was an involved step, so the same hard-coded process
     will not work with another dataset.

### Internet References:

[[https://intoli.com/blog/pca-and-svd/]{.underline}](https://intoli.com/blog/pca-and-svd/)

Eigenvalues vs SVD Decomposition
--------------------------------

Calculating principal components (PCs) using eigenvalue decomposition
and SVD yielded similar results. Though they are not exactly the same,
they are related in that the eigenvalues from eigen decomposition are
equal to the square of the singular values resulting from SVD, scaled
within the range 0:N-1, where N is the size of the input. Printing out
the result of this relationship, once computed, is:

  --------------------------------------------------------------------------
  ~~~
  PCA:\
  \[52.43562789 27.0014608 18.13407969 11.75436066 10.32441643 8.75238355\
  0.5530527 7.37434293 3.78288767 5.79044214 5.0687423 \]\
  SVD:\
  \[52.43562789 27.0014608 18.13407969 11.75436066 10.32441643 8.75238355\
  7.37434293 5.79044214 5.0687423 3.78288767 0.5530527 \]
~~~
  --------------------------------------------------------------------------

As you can see, they are the same. The original:

  --------------------------------------------------------------------------
  ~~~
  PCA:\
  \[7.12304423e+00 1.88880540e+00 8.51929654e-01 3.57940400e-01\
  2.76149157e-01 1.98456523e-01 7.92402290e-04 1.40883248e-01\
  3.70731584e-02 8.68632648e-02 6.65599703e-02\]\
  SVD:\
  \[52.43562789 27.0014608 18.13407969 11.75436066 10.32441643 8.75238355\
  7.37434293 5.79044214 5.0687423 3.78288767 0.5530527 \]
  ~~~
  --------------------------------------------------------------------------

### Finding and Comparing Principal Components

The corresponding eigen*vectors*, and the columns of the second unit
vector given by SVD are the principal components we are looking for. By
printing them out we can see that they are similar:

  -------------------------------------------
  ~~~
  First two Principal Components from PCA:\
  \[\[ 0.26375044 0.4685087 \]\
  \[ 0.26231864 0.47014659\]\
  \[ 0.34708049 -0.01534719\]\
  \[ 0.33418876 0.07803201\]\
  \[ 0.31860226 0.29221348\]\
  \[-0.31048173 -0.00336594\]\
  \[-0.30658864 -0.01096446\]\
  \[ 0.33632937 -0.16746357\]\
  \[ 0.26621003 -0.41817711\]\
  \[ 0.25679019 -0.40841138\]\
  \[ 0.29605459 -0.31289135\]\]
  ~~~
  -------------------------------------------

  -------------------------------------------
  ~~~
  First two Principal Components from SVD:\
  \[\[-0.26375044 -0.4685087 \]\
  \[-0.26231864 -0.47014659\]\
  \[-0.34708049 0.01534719\]\
  \[-0.33418876 -0.07803201\]\
  \[-0.31860226 -0.29221348\]\
  \[ 0.31048173 0.00336594\]\
  \[ 0.30658864 0.01096446\]\
  \[-0.33632937 0.16746357\]\
  \[-0.26621003 0.41817711\]\
  \[-0.25679019 0.40841138\]\
  \[-0.29605459 0.31289135\]\]
  ~~~
  -------------------------------------------

The proportions between the two are exactly the same, though one is the
inverse of the other (the columns are the PCs).

Plotting the first two PCs from SVD yield us the next two plots. Note
that although they look the same, they are only acutely similar. A
printout of each PC shows that they are in fact distinct:

  -------------------------------------------
  ~~~
  First two Principal Components from SVD:\
  \[\[ 0.26375044 0.4685087 \]\
  \[ 0.26231864 0.47014659\]\
  \[ 0.34708049 -0.01534719\]\
  \[ 0.33418876 0.07803201\]\
  \[ 0.31860226 0.29221348\]\
  \[-0.31048173 -0.00336594\]\
  \[-0.30658864 -0.01096446\]\
  \[ 0.33632937 -0.16746357\]\
  \[ 0.26621003 -0.41817711\]\
  \[ 0.25679019 -0.40841138\]\
  \[ 0.29605459 -0.31289135\]\]
  ~~~
  -------------------------------------------

![](.//media/image1.png){width="3.4743055555555555in"
height="2.6659722222222224in"}![](.//media/image2.png){width="3.125in"
height="2.5034722222222223in"}

*SVD PC results (in blue). Original data distribution shown in black. I
achieved the one-dimensional projection by projecting onto a single PC
multiplied by the identity matrix.*

### Projecting onto Principal Components in One Dimension

Projecting our data onto our first principal component for each shows us
that, although the two do have a defined mathematical proportion, they
are not equal.

![](.//media/image3.png){width="4.291666666666667in"
height="3.423611111111111in"}

### Projecting onto Principal Components in Two Dimensions

Projecting our data onto our first two PC's gives us the following
scatterplot:

![](.//media/image4.png){width="4.838888888888889in" height="3.68125in"}

It appeared that the PCA and SVD function outputs are inverse of each
other. Sure enough, multiplying one by (-1) yields two identical plots:

![](.//media/image5.png){width="3.1166666666666667in"
height="2.411111111111111in"}![](.//media/image6.png){width="3.088888888888889in"
height="2.3805555555555555in"}

This tells us that between the two methods, the relationships between
the data on the first two principal components are the same, only one is
negative and the other positive. The first two principal components
given by SVD give us the two axes on which the data is most widely
represented. This is commonly referred to as having the most *variance*.
These axes are combinations of existing features set at different
scales, as some of the features are far more important than others. When
combined, this allows us to represent our previously 11-dimensional data
in only two dimensions, with nearly the same level of uniqueness between
points. If we were to simply select two features and plot those as our
new data points, many of the data points would appear similar or the
same, even if they vary among the other 9 unrecorded features.
