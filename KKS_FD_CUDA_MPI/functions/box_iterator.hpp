#ifndef __CUFFTMP_BOX_ITERATOR_HPP__
#define __CUFFTMP_BOX_ITERATOR_HPP__

#ifndef ENABLE_CUFFTMP
#define ENABLE_CUFFTMP 0
#endif

#if ENABLE_CUFFTMP == 1

#include <iterator>
#include <cstddef>
#include <tuple>

/**
 * This iterator lets one iterate through the underlying data
 * associated to a (lower, upper, strides) box, and exposes the mapping
 * between global 3D coordinates (x, y, z) and local linear
 * indices.
 *
 * This iterator can be used in host code
 */

using int64 = long long int;

struct Box3D {
    int64 lower[3];
    int64 upper[3];
    int64 strides[3];
};

template<typename T>
struct BoxIterator
{
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

    BoxIterator(int64 i, Box3D box, T* ptr) : i_(i), box_(box), ptr_(ptr),
                                                    lx_(box.upper[0] - box.lower[0]),
                                                    ly_(box.upper[1] - box.lower[1]),
                                                    lz_(box.upper[2] - box.lower[2]) {
        linear_to_box3d(i_, &x_, &y_, &z_);
    };

    BoxIterator& operator++() { increment(1); return *this; }

    BoxIterator operator++(int) {
        BoxIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    BoxIterator& operator--() { increment(-1); return *this; }

    BoxIterator operator--(int) {
        BoxIterator tmp = *this;
        --(*this);
        return tmp;
    }

    BoxIterator& operator+=(difference_type rhs) { increment(rhs); return *this; }

    BoxIterator& operator-=(difference_type rhs) { increment(-rhs); return *this; }


    reference operator*() const { return ptr_[i()]; }

    pointer operator->() { return ptr_ + i(); }

    BoxIterator& operator[](difference_type rhs) const { return (*this + rhs); }

    friend difference_type operator-(const BoxIterator& a, const BoxIterator& b) {return a.i_ - b.i_; }

    friend BoxIterator operator-(const BoxIterator& a, difference_type n) { a -= n; return a; }

    friend BoxIterator operator+(const BoxIterator& a, difference_type n) { a += n; return a; }

    friend BoxIterator operator+(difference_type n, const BoxIterator& a) { return a+n; }

    friend bool operator==(const BoxIterator& a, const BoxIterator& b) { return a.i_ == b.i_; }

    friend bool operator!=(const BoxIterator& a, const BoxIterator& b) { return a.i_ != b.i_; }

    friend bool operator>(const BoxIterator& a, const BoxIterator& b) { return a.i_ > b.i_; }

    friend bool operator<(const BoxIterator& a, const BoxIterator& b) { return a.i_ < b.i_; }

    friend bool operator>=(const BoxIterator& a, const BoxIterator& b) { return a.i_ >= b.i_; }

    friend bool operator<=(const BoxIterator& a, const BoxIterator& b) { return a.i_ <= b.i_; }

    /**
     * Return the global X coordinate of the iterator
     */
    int64 x() const { return x_; }

    /**
     * Return the global Y coordinate of the iterator
     */
    int64 y() const { return y_; }

    /**
     * Return the global Z coordinate of the iterator
     */
    int64 z() const { return z_; }

    /**
     * Return the linear position of the iterator
     * in the local data buffer
     */
    int64 i() const {
        return (x_ - box_.lower[0]) * box_.strides[0] + (y_ - box_.lower[1]) * box_.strides[1] + (z_ - box_.lower[2]) * box_.strides[2];
    }

private:

    // Current 3D global index in the box
    int64 x_, y_, z_;
    // Current linear 3D index (not the location in memory)
    int64 i_;
    // Global box lower and upper corner and local strides
    const Box3D box_;
    // Underlying data pointer
    T* ptr_;
    // Length of the X, Y and Z dimensions
    const int64 lx_, ly_, lz_;

    // Linear to 3D coordinates
    void linear_to_box3d(int64 i, int64* x, int64* y, int64* z) {
        if(lx_ * ly_ * lz_ > 0) {
            *x  =   i  / (ly_ * lz_);
            i  -= (*x) * (ly_ * lz_);
            *y  =   i  / (lz_);
            i  -= (*y) * (lz_);
            *z  =   i;
        } else {
            *x = 0;
            *y = 0;
            *z = 0;
        }
        *x += box_.lower[0];
        *y += box_.lower[1];
        *z += box_.lower[2];
    }

    // Increment/decrement by n
    void increment(difference_type n) {
        i_ += n;
        linear_to_box3d(i_, &x_, &y_, &z_);
    }

};

template<typename T>
BoxIterator<T> BoxIteratorBegin(Box3D box, T* ptr) {
    return BoxIterator<T>(0, box, ptr);
};

template<typename T>
BoxIterator<T> BoxIteratorEnd(Box3D box, T* ptr) {
    return BoxIterator<T>( (box.upper[0] - box.lower[0]) * (box.upper[1] - box.lower[1]) * (box.upper[2] - box.lower[2]), box, ptr);
};

template<typename T>
std::pair<BoxIterator<T>,BoxIterator<T>> BoxIterators(Box3D box, T* ptr) {
    return {BoxIteratorBegin<T>(box, ptr),BoxIteratorEnd<T>(box, ptr)};
}

int64 slabs_displacement(int64 length, int rank, int size) {
    int ranks_cutoff = length % size;
    return (rank < ranks_cutoff ? rank * (length / size + 1) : ranks_cutoff * (length / size + 1) + (rank - ranks_cutoff) * (length / size));
}

Box3D buildBox3D(int rank, int size, int64 nx, int64 ny, int64 nz) {
    int64 x_start      = slabs_displacement(nx, rank,   size);
    int64 x_end        = slabs_displacement(nx, rank+1, size);
    int64 my_ny        = ny;
    int64 my_nz        = nz;
    return {
        {x_start, 0, 0}, {x_end, my_ny, my_nz}, {my_ny * my_nz, my_nz, 1}
    };
}

template<typename T>
BoxIterator<T> BoxIteratorBegin(int rank, int size, int64 nx, int64 ny, int64 nz, T* ptr) {
    Box3D box = buildBox3D(rank, size, nx, ny, nz);
    return BoxIteratorBegin<T>(box, ptr);
}

template<typename T>
BoxIterator<T> BoxIteratorEnd(int rank, int size, int64 nx, int64 ny, int64 nz, T* ptr) {
    Box3D box = buildBox3D(rank, size, nx, ny, nz);
    return BoxIteratorEnd<T>(box, ptr);
}

template<typename T>
std::pair<BoxIterator<T>,BoxIterator<T>> BoxIterators(int rank, int size, int64 nx, int64 ny, int64 nz, T* ptr) {
    return {BoxIteratorBegin<T>(rank, size, nx, ny, nz, ptr),BoxIteratorEnd<T>(rank, size, nx, ny, nz, ptr)};
}

#endif // ENABLE_CUFFTMP
#endif // __CUFFTMP_BOX_ITERATOR_HPP__
