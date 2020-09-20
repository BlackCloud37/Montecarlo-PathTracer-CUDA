//
// Created by blackcloud on 2020/5/7.
//

#ifndef CUDA_TEST_VEC_MAT_CUH
#define CUDA_TEST_VEC_MAT_CUH
#include "helper_math.h"
#define Vector4f float4
#define Vector3f float3
#define Vector2f float2


__device__ __host__ static inline float getDimension(const float3& v, const uint d) {
    return d == 0 ? v.x : (d == 1 ? v.y : v.z);
}


__device__ __host__ static inline float getDimension(const float4& v, const uint d) {
    return d == 0 ? v.x : (d == 1 ? v.y : (d == 2 ? v.z : v.w));
}


__device__ __host__ float determinant3x3( float m00, float m01, float m02,
                                          float m10, float m11, float m12,
                                          float m20, float m21, float m22 ) {
    return
            (
                    m00 * ( m11 * m22 - m12 * m21 )
                    - m01 * ( m10 * m22 - m12 * m20 )
                    + m02 * ( m10 * m21 - m11 * m20 )
            );
}
__device__ __host__ float determinant2x2( float m00, float m01,
                                float m10, float m11 )
{
    return( m00 * m11 - m01 * m10 );
}

struct Matrix3f {
    float m_elements[9];
    __device__ __host__ Matrix3f( float fill=0.f )
    {
        for( int i = 0; i < 9; ++i )
        {
            m_elements[ i ] = fill;
        }
    }

    __device__ __host__ Matrix3f( float m00, float m01, float m02,
                        float m10, float m11, float m12,
                        float m20, float m21, float m22 )
    {
        m_elements[ 0 ] = m00;
        m_elements[ 1 ] = m10;
        m_elements[ 2 ] = m20;

        m_elements[ 3 ] = m01;
        m_elements[ 4 ] = m11;
        m_elements[ 5 ] = m21;

        m_elements[ 6 ] = m02;
        m_elements[ 7 ] = m12;
        m_elements[ 8 ] = m22;
    }
    __device__ __host__ Matrix3f( const Matrix3f& rm )
    {
        memcpy( m_elements, rm.m_elements, 9 * sizeof( float ) );
    }

    __device__ __host__ const float& operator () ( int i, int j ) const
    {
        return m_elements[ j * 3 + i ];
    }

    __device__ __host__ float& operator () ( int i, int j )
    {
        return m_elements[ j * 3 + i ];
    }
    __device__ __host__ Matrix3f inverse( bool* pbIsSingular = nullptr, float epsilon = 0.f) const
    {
        float m00 = m_elements[ 0 ];
        float m10 = m_elements[ 1 ];
        float m20 = m_elements[ 2 ];

        float m01 = m_elements[ 3 ];
        float m11 = m_elements[ 4 ];
        float m21 = m_elements[ 5 ];

        float m02 = m_elements[ 6 ];
        float m12 = m_elements[ 7 ];
        float m22 = m_elements[ 8 ];

        float cofactor00 =  determinant2x2( m11, m12, m21, m22 );
        float cofactor01 = -determinant2x2( m10, m12, m20, m22 );
        float cofactor02 =  determinant2x2( m10, m11, m20, m21 );

        float cofactor10 = -determinant2x2( m01, m02, m21, m22 );
        float cofactor11 =  determinant2x2( m00, m02, m20, m22 );
        float cofactor12 = -determinant2x2( m00, m01, m20, m21 );

        float cofactor20 =  determinant2x2( m01, m02, m11, m12 );
        float cofactor21 = -determinant2x2( m00, m02, m10, m12 );
        float cofactor22 =  determinant2x2( m00, m01, m10, m11 );

        float determinant = m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02;

        bool isSingular = ( fabs( determinant ) < epsilon );
        if( isSingular )
        {
            if( pbIsSingular != NULL )
            {
                *pbIsSingular = true;
            }
            return Matrix3f();
        }
        else
        {
            if( pbIsSingular != NULL )
            {
                *pbIsSingular = false;
            }

            float reciprocalDeterminant = 1.0f / determinant;

            return Matrix3f
                    (
                            cofactor00 * reciprocalDeterminant, cofactor10 * reciprocalDeterminant, cofactor20 * reciprocalDeterminant,
                            cofactor01 * reciprocalDeterminant, cofactor11 * reciprocalDeterminant, cofactor21 * reciprocalDeterminant,
                            cofactor02 * reciprocalDeterminant, cofactor12 * reciprocalDeterminant, cofactor22 * reciprocalDeterminant
                    );
        }
    }
    // static
    static __device__ Matrix3f rotateY( float radians )
    {
        float c = cos( radians );
        float s = sin( radians );

        return Matrix3f
                (
                        c, 0, s,
                        0, 1, 0,
                        -s, 0, c
                );
    }


};



__device__ Vector3f operator * ( const Matrix3f& m, const Vector3f& v )
{
    float output[3] = {0,0,0};
    float vv[3] = {v.x, v.y, v.z};
    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            output[ i ] += m( i, j ) * vv[ j ];
        }
    }

    return make_float3(output[0],output[1],output[2]);
}

__device__ Matrix3f operator * ( const Matrix3f& x, const Matrix3f& y )
{
    Matrix3f product; // zeroes

    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            for( int k = 0; k < 3; ++k )
            {
                product( i, k ) += x( i, j ) * y( j, k );
            }
        }
    }

    return product;
}
struct Matrix4f {
    float m_elements[16];
    __device__ __host__ Matrix4f( float fill = 0.f) {
        for( int i = 0; i < 16; ++i ) {
            m_elements[ i ] = fill;
        }
    }
    __device__ __host__ Matrix4f(const Matrix4f& m) {
        for( int i = 0; i < 16; ++i ) {
            m_elements[ i ] = m.m_elements[i];
        }
    }
    __device__ __host__ Matrix4f( float m00, float m01, float m02, float m03,
                                  float m10, float m11, float m12, float m13,
                                  float m20, float m21, float m22, float m23,
                                  float m30, float m31, float m32, float m33 )
    {
        m_elements[ 0 ] = m00;
        m_elements[ 1 ] = m10;
        m_elements[ 2 ] = m20;
        m_elements[ 3 ] = m30;

        m_elements[ 4 ] = m01;
        m_elements[ 5 ] = m11;
        m_elements[ 6 ] = m21;
        m_elements[ 7 ] = m31;

        m_elements[ 8 ] = m02;
        m_elements[ 9 ] = m12;
        m_elements[ 10 ] = m22;
        m_elements[ 11 ] = m32;

        m_elements[ 12 ] = m03;
        m_elements[ 13 ] = m13;
        m_elements[ 14 ] = m23;
        m_elements[ 15 ] = m33;
    }

    __device__ __host__ const float& operator () (int i, int j) const {
        return m_elements[j*4 + i];
    }
    __device__ __host__ float& operator () (int i, int j) {
        return m_elements[j*4 + i];
    }
    __device__ __host__ Matrix4f inverse(bool* pbIsSingular = nullptr, float epsilon = 0.f) const {
        float m00 = m_elements[ 0 ];
        float m10 = m_elements[ 1 ];
        float m20 = m_elements[ 2 ];
        float m30 = m_elements[ 3 ];

        float m01 = m_elements[ 4 ];
        float m11 = m_elements[ 5 ];
        float m21 = m_elements[ 6 ];
        float m31 = m_elements[ 7 ];

        float m02 = m_elements[ 8 ];
        float m12 = m_elements[ 9 ];
        float m22 = m_elements[ 10 ];
        float m32 = m_elements[ 11 ];

        float m03 = m_elements[ 12 ];
        float m13 = m_elements[ 13 ];
        float m23 = m_elements[ 14 ];
        float m33 = m_elements[ 15 ];

        float cofactor00 =  determinant3x3( m11, m12, m13, m21, m22, m23, m31, m32, m33 );
        float cofactor01 = -determinant3x3( m12, m13, m10, m22, m23, m20, m32, m33, m30 );
        float cofactor02 =  determinant3x3( m13, m10, m11, m23, m20, m21, m33, m30, m31 );
        float cofactor03 = -determinant3x3( m10, m11, m12, m20, m21, m22, m30, m31, m32 );

        float cofactor10 = -determinant3x3( m21, m22, m23, m31, m32, m33, m01, m02, m03 );
        float cofactor11 =  determinant3x3( m22, m23, m20, m32, m33, m30, m02, m03, m00 );
        float cofactor12 = -determinant3x3( m23, m20, m21, m33, m30, m31, m03, m00, m01 );
        float cofactor13 =  determinant3x3( m20, m21, m22, m30, m31, m32, m00, m01, m02 );

        float cofactor20 =  determinant3x3( m31, m32, m33, m01, m02, m03, m11, m12, m13 );
        float cofactor21 = -determinant3x3( m32, m33, m30, m02, m03, m00, m12, m13, m10 );
        float cofactor22 =  determinant3x3( m33, m30, m31, m03, m00, m01, m13, m10, m11 );
        float cofactor23 = -determinant3x3( m30, m31, m32, m00, m01, m02, m10, m11, m12 );

        float cofactor30 = -determinant3x3( m01, m02, m03, m11, m12, m13, m21, m22, m23 );
        float cofactor31 =  determinant3x3( m02, m03, m00, m12, m13, m10, m22, m23, m20 );
        float cofactor32 = -determinant3x3( m03, m00, m01, m13, m10, m11, m23, m20, m21 );
        float cofactor33 =  determinant3x3( m00, m01, m02, m10, m11, m12, m20, m21, m22 );

        float determinant = m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03;

        bool isSingular = ( fabs( determinant ) < epsilon );
        if( isSingular )
        {
            if( pbIsSingular != NULL )
            {
                *pbIsSingular = true;
            }
            return Matrix4f();
        }
        else
        {
            if( pbIsSingular != NULL )
            {
                *pbIsSingular = false;
            }

            float reciprocalDeterminant = 1.0f / determinant;

            return Matrix4f
                    (
                            cofactor00 * reciprocalDeterminant, cofactor10 * reciprocalDeterminant, cofactor20 * reciprocalDeterminant, cofactor30 * reciprocalDeterminant,
                            cofactor01 * reciprocalDeterminant, cofactor11 * reciprocalDeterminant, cofactor21 * reciprocalDeterminant, cofactor31 * reciprocalDeterminant,
                            cofactor02 * reciprocalDeterminant, cofactor12 * reciprocalDeterminant, cofactor22 * reciprocalDeterminant, cofactor32 * reciprocalDeterminant,
                            cofactor03 * reciprocalDeterminant, cofactor13 * reciprocalDeterminant, cofactor23 * reciprocalDeterminant, cofactor33 * reciprocalDeterminant
                    );
        }
    }
    __host__ static Matrix4f translation(float x, float y, float z) {
        return Matrix4f
                (
                        1, 0, 0, x,
                        0, 1, 0, y,
                        0, 0, 1, z,
                        0, 0, 0, 1
                );
    }
    __host__ static Matrix4f scaling(float sx, float sy, float sz) {
        return Matrix4f
                (
                        sx, 0, 0, 0,
                        0, sy, 0, 0,
                        0, 0, sz, 0,
                        0, 0, 0, 1
                );
    }
    __host__ static Matrix4f uniformScaling( float s )
    {
        return Matrix4f
                (
                        s, 0, 0, 0,
                        0, s, 0, 0,
                        0, 0, s, 0,
                        0, 0, 0, 1
                );
    }
    __host__ static Matrix4f identity()
    {
        Matrix4f m(0);

        m( 0, 0 ) = 1;
        m( 1, 1 ) = 1;
        m( 2, 2 ) = 1;
        m( 3, 3 ) = 1;

        return m;
    }
    __host__ static Matrix4f rotateX( float radians )
    {
        float c = cos( radians );
        float s = sin( radians );

        return Matrix4f
                (
                        1, 0, 0, 0,
                        0, c, -s, 0,
                        0, s, c, 0,
                        0, 0, 0, 1
                );
    }

    __host__ static Matrix4f rotateY( float radians )
    {
        float c = cos( radians );
        float s = sin( radians );

        return Matrix4f
                (
                        c, 0, s, 0,
                        0, 1, 0, 0,
                        -s, 0, c, 0,
                        0, 0, 0, 1
                );
    }
    __host__ static Matrix4f rotateZ( float radians )
    {
        float c = cos( radians );
        float s = sin( radians );

        return Matrix4f
                (
                        c, -s, 0, 0,
                        s, c, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1
                );
    }
    __host__ static Matrix4f rotation( const Vector3f& rDirection, float radians )
    {
        Vector3f normalizedDirection = normalize(rDirection);

        float cosTheta = cos( radians );
        float sinTheta = sin( radians );

        float x = normalizedDirection.x;
        float y = normalizedDirection.y;
        float z = normalizedDirection.z;

        return Matrix4f
                (
                        x * x * ( 1.0f - cosTheta ) + cosTheta,			y * x * ( 1.0f - cosTheta ) - z * sinTheta,		z * x * ( 1.0f - cosTheta ) + y * sinTheta,		0.0f,
                        x * y * ( 1.0f - cosTheta ) + z * sinTheta,		y * y * ( 1.0f - cosTheta ) + cosTheta,			z * y * ( 1.0f - cosTheta ) - x * sinTheta,		0.0f,
                        x * z * ( 1.0f - cosTheta ) - y * sinTheta,		y * z * ( 1.0f - cosTheta ) + x * sinTheta,		z * z * ( 1.0f - cosTheta ) + cosTheta,			0.0f,
                        0.0f,											0.0f,											0.0f,											1.0f
                );
    }

    __host__ __device__ Matrix4f transposed() const
    {
        Matrix4f out;
        for( int i = 0; i < 4; ++i )
        {
            for( int j = 0; j < 4; ++j )
            {
                out( j, i ) = ( *this )( i, j );
            }
        }
        return out;
    }

};

__device__ __host__ Matrix4f operator * ( const Matrix4f& x, const Matrix4f& y )
{
    Matrix4f product = Matrix4f(0); // zeroes

    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            for( int k = 0; k < 4; ++k )
            {
                product( i, k ) += x( i, j ) * y( j, k );
            }
        }
    }

    return product;
}

__device__ __host__ Vector4f operator * (const Matrix4f& m, const Vector4f& v) {
    float output[4] = {0,0,0,0};
    //Vector4f output = make_float4(0,0,0,0);
    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            output[i] += m(i,j) * getDimension(v, j);
        }
    }
    return make_float4(output[0], output[1], output[2], output[3]);
}


// transforms a 3D point using a matrix, returning a 3D point
__device__ __host__  static Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point) {
    auto f4 = mat * make_float4(point, 1);
    return make_float3(f4.x, f4.y, f4.z);
}

// transform a 3D directino using a matrix, returning a direction
__device__ __host__ static Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir) {
    auto f4 = mat * make_float4(dir, 0);
    return make_float3(f4.x, f4.y, f4.z);
}


#endif //CUDA_TEST_VEC_MAT_CUH
