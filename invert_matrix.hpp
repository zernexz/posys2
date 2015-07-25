/*
The following code inverts the matrix input using LU-decomposition with backsubstitution of unit vectors. Reference: Numerical Recipies in C, 2nd ed., by Press, Teukolsky, Vetterling & Flannery.

you can solve Ax=b using three lines of ublas code:

permutation_matrix<> piv;
lu_factorize(A, piv);
lu_substitute(A, piv, x);

*/
 #ifndef INVERT_MATRIX_HPP
 #define INVERT_MATRIX_HPP

 // REMEMBER to update "lu.hpp" header includes from boost-CVS
 #include <boost/numeric/ublas/vector.hpp>
 #include <boost/numeric/ublas/vector_proxy.hpp>
 #include <boost/numeric/ublas/matrix.hpp>
 #include <boost/numeric/ublas/triangular.hpp>
 #include <boost/numeric/ublas/lu.hpp>
 #include <boost/numeric/ublas/io.hpp>

namespace bnu = boost::numeric::ublas;

/* Matrix inversion routine.
Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
bool InvertMatrix(const boost::numeric::ublas::matrix<T>& input, boost::numeric::ublas::matrix<T>& inverse)
{
   typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;

   // create a working copy of the input
   boost::numeric::ublas::matrix<T> A(input);

   // create a permutation matrix for the LU-factorization
   pmatrix pm(A.size1());

   // perform LU-factorization
   int res = boost::numeric::ublas::lu_factorize(A, pm);
   if (res != 0)
       return false;

   // create identity matrix of "inverse"
   inverse.assign(boost::numeric::ublas::identity_matrix<T> (A.size1()));

   // backsubstitute to get the inverse
   boost::numeric::ublas::lu_substitute(A, pm, inverse);

   return true;
}


 
int determinant_sign(const bnu::permutation_matrix<std ::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}
 
double determinant( bnu::matrix<double>& m ) {
    bnu::permutation_matrix<std ::size_t> pm(m.size1());
    double det = 1.0;
    if( bnu::lu_factorize(m,pm) ) {
        det = 0.0;
    } else {
        for(int i = 0; i < m.size1(); i++)
            det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
    }
    return det;
}

 #endif //INVERT_MATRIX_HPP
