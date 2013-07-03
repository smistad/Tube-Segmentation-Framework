#include "eigenanalysisOfHessian.hpp"
#include "SIPL/Types.hpp"
using namespace SIPL;
#define MAX(a,b) a > b ? a : b



#define SIZE 3

float hypot2(float x, float y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

void tred2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
  }

  // Householder reduction to tridiagonal form.

  for (int i = SIZE-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    float scale = 0.0f;
    float h = 0.0f;
    for (int k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0f) {
      e[i] = d[i-1];
      for (int j = 0; j < i; j++) {
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
        V[j][i] = 0.0f;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      float f = d[i-1];
      float g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e[j] = 0.0f;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0f;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      float hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i-1; k++) {
          V[k][j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1][j];
        V[i][j] = 0.0f;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.

  for (int i = 0; i < SIZE-1; i++) {
    V[SIZE-1][i] = V[i][i];
    V[i][i] = 1.0f;
    float h = d[i+1];
    if (h != 0.0f) {
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i+1] / h;
      }
      for (int j = 0; j <= i; j++) {
        float g = 0.0f;
        for (int k = 0; k <= i; k++) {
          g += V[k][i+1] * V[k][j];
        }
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V[k][i+1] = 0.0f;
    }
  }
  for (int j = 0; j < SIZE; j++) {
    d[j] = V[SIZE-1][j];
    V[SIZE-1][j] = 0.0f;
  }
  V[SIZE-1][SIZE-1] = 1.0f;
  e[0] = 0.0f;
}

// Symmetric tridiagonal QL algorithm.

void tql2(float V[SIZE][SIZE], float d[SIZE], float e[SIZE]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int i = 1; i < SIZE; i++) {
    e[i-1] = e[i];
  }
  e[SIZE-1] = 0.0f;

  float f = 0.0f;
  float tst1 = 0.0f;
  float eps = pow(2.0f,-52.0f);
  for (int l = 0; l < SIZE; l++) {

    // Find small subdiagonal element

    tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < SIZE) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        float g = d[l];
        float p = (d[l+1] - g) / (2.0f * e[l]);
        float r = hypot2(p,1.0f);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        float dl1 = d[l+1];
        float h = g - d[l];
        for (int i = l+2; i < SIZE; i++) {
          d[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d[m];
        float c = 1.0f;
        float c2 = c;
        float c3 = c;
        float el1 = e[l+1];
        float s = 0.0f;
        float s2 = 0.0f;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);

          // Accumulate transformation.

          for (int k = 0; k < SIZE; k++) {
            h = V[k][i+1];
            V[k][i+1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;

        // Check for convergence.

      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0f;
  }

  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < SIZE-1; i++) {
    int k = i;
    float p = d[i];
    for (int j = i+1; j < SIZE; j++) {
      if (fabs(d[j]) < fabs(p)) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < SIZE; j++) {
        p = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = p;
      }
    }
  }
}

void eigen_decomposition(float A[SIZE][SIZE], float V[SIZE][SIZE], float d[SIZE]) {
  float e[SIZE];
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      V[i][j] = A[i][j];
    }
  }
  tred2(V, d, e);
  tql2(V, d, e);
}


#define POS(pos) pos.x+pos.y*size.x+pos.z*size.x*size.y
SIPL::float3 gradient(TubeSegmentation &TS, SIPL::int3 pos, int volumeComponent, int dimensions, int3 size) {
    float * Fx = TS.Fx;
    float * Fy = TS.Fy;
    float * Fz = TS.Fz;
    float f100, f_100, f010, f0_10, f001, f00_1;
    SIPL::int3 npos = pos;
    switch(volumeComponent) {
        case 0:

        npos.x +=1;
        f100 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fx[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
        case 1:

        npos.x +=1;
        f100 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fy[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
        case 2:

        npos.x +=1;
        f100 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        npos.x -=2;
        f_100 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        if(dimensions > 1) {
            npos = pos;
            npos.y += 1;
            f010 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.y -= 2;
            f0_10 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
        if(dimensions > 2) {
            npos = pos;
            npos.z += 1;
            f001 = Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
            npos.z -= 2;
            f00_1 =Fz[POS(npos)]/sqrt(Fx[POS(npos)]*Fx[POS(npos)]+Fy[POS(npos)]*Fy[POS(npos)]+Fz[POS(npos)]*Fz[POS(npos)]);
        }
    break;
    }

    float3 grad(0.5f*(f100-f_100), 0.5f*(f010-f0_10), 0.5f*(f001-f00_1));


    return grad;
}


float3 getTubeDirection(TubeSegmentation &T, int3 pos, int3 size) {

    // Do gradient on Fx, Fy and Fz and normalization
    float3 Fx = gradient(T, pos,0,1,size);
    float3 Fy = gradient(T, pos,1,2,size);
    float3 Fz = gradient(T, pos,2,3,size);

    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    float3 e1(eigenVectors[0][0], eigenVectors[1][0], eigenVectors[2][0]);
    return e1;
}

void doEigen(TubeSegmentation &T, int3 pos, int3 size, float3 * lambda, float3 * e1, float3 * e2, float3 * e3) {

    // Do gradient on Fx, Fy and Fz and normalization
    float3 Fx = gradient(T, pos,0,1,size);
    float3 Fy = gradient(T, pos,1,2,size);
    float3 Fz = gradient(T, pos,2,3,size);

    float Hessian[3][3] = {
        {Fx.x, Fy.x, Fz.x},
        {Fy.x, Fy.y, Fz.y},
        {Fz.x, Fz.y, Fz.z}
    };
    float eigenValues[3];
    float eigenVectors[3][3];
    eigen_decomposition(Hessian, eigenVectors, eigenValues);
    e1->x = eigenVectors[0][0];
    e1->y = eigenVectors[1][0];
    e1->z = eigenVectors[2][0];
    e2->x = eigenVectors[0][1];
    e2->y = eigenVectors[1][1];
    e2->z = eigenVectors[2][1];
    e3->x = eigenVectors[0][2];
    e3->y = eigenVectors[1][2];
    e3->z = eigenVectors[2][2];
    lambda->x = eigenValues[0];
    lambda->y = eigenValues[1];
    lambda->z = eigenValues[2];
}
