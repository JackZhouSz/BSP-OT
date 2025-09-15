#ifndef PERIODIC2DGRID_H
#define PERIODIC2DGRID_H

#include "types.h"


template<class T>
class Grid
{
private:
    long width;
    long height;

    //Eigen::Vector<T,-1> values;
    Vec values;

    static Index mod(Index i,long n){
        while (i < 0)
            i += n;
        while (i >= n)
            i -= n;
        return i;
    }

    bool periodic = true;
public:

    using grid_fill_function = std::function<T(Index,Index)>;
    using grid_function = std::function<T(Index,Index)>;

    Grid(){}

    Grid(long _width,long _height,T fill = T()){
        allocate(_width,_height,fill);
    }
    Grid(long _width,long _height,Vec X){
        allocate(_width,_height,T());
        setFromVector(X);
    }

    void allocate(long _width,long _height,T fill = T()){
        width = _width;
        height = _height;
        values = Vec::Ones(width*height)*fill;
    }

    void fill(const grid_fill_function& f){
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                at(i,j) = f(i,j);
    }

    Index getIndex(Index i,Index j) const {
        return mod(j,height)*width + mod(i,width);
    }

    Index getIndex(grid_Index I) const {
        return getIndex(I.first,I.second);
    }

    T& at(Index i,Index j) {
        return values[getIndex(i,j)];
    }

    T at(Index i,Index j) const {
        if (!periodic)
            if (check(i,j))
                return T();
        return values[getIndex(i,j)];
    }

    bool check(Index i,Index j) const {
        return i < 0
                || i >= width
                || j < 0
                || j >= height;
    }

    const Vec& data() const {
        return values;
    }

    grid_Index toGridIndex(Index i) const{
        return {i%width,i/width};
    }

    std::array<Index,4> getNeighbors(Index i) const {
        return getNeighbors(toGridIndex(i));
    }

    std::array<Index,4> getNeighbors(grid_Index I) const {
        Index i = I.first;
        Index j = I.second;
        return {
            getIndex(i,j-1),
            getIndex(i+1,j),
            getIndex(i,j+1),
            getIndex(i-1,j)
        };
    }
    std::vector<Index> getNonPeriodicNeighbors(Index I) const {
        return getNonPeriodicNeighbors(toGridIndex(I));
    }
    std::vector<Index> getNonPeriodicNeighbors(grid_Index I) const {
        return getNonPeriodicNeighbors(I.first,I.second);
    }

    std::vector<Index> getNonPeriodicNeighbors(Index i,Index j) const {
        std::vector<Index> N;
        /*
        for (int dx = -1;dx <= 1;dx++){
            //check 8 neighbors
            for (int dy = -1;dy <= 1;dy++){
                if (dx == 0 && dy == 0)
                    continue;
                if (check(i+dx,j+dy))
                    continue;
                N.push_back(getIndex(i+dx,j+dy));
            }
        }
        return N;
*/
        if (!check(i+1,j))
            N.push_back(getIndex(i+1,j));
        if (!check(i-1,j))
            N.push_back(getIndex(i-1,j));
        if (!check(i,j+1))
            N.push_back(getIndex(i,j+1));
        if (!check(i,j-1))
            N.push_back(getIndex(i,j-1));
        return N;
    }


    Vec vectorize() const
    {
        Vec X(width*height);
        for (Index i = 0;i<width*height;i++)
            X(i) = values[i];
        return X;
    }

    void setFromVector(const Vec& X) {
        values = X;
    }

    Grid operator+(const Grid& other) const {
        Grid S(width,height);
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                S.at(i,j) = at(i,j) + other.at(i,j);
        return S;
    }

    void operator+=(const Grid& other){
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                at(i,j) += other.at(i,j);
    }


    Grid operator-(const Grid& other) const {
        Grid S(width,height);
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                S.at(i,j) = at(i,j) - other.at(i,j);
        return S;
    }

    Grid operator*(scalar x) const {
        Grid S(width,height);
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                S.at(i,j) = at(i,j)*x;
        return S;
    }
    Grid operator/(scalar x) const {
        Grid S(width,height);
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                S.at(i,j) = at(i,j)/x;
        return S;
    }

    template<class T2>
    Grid<T2> convert(const std::function<T2()>& converter)
    {
        Grid S(width,height);
        for (Index j = 0;j<height;j++)
            for (Index i = 0;i<width;i++)
                S.at(i,j) = converter(at(i,j));
        return S;
    }
    static scalar dist(const grid_Index& I,const grid_Index& I2) {
        scalar dx = I.first - I2.first;
        scalar dy = I.second - I2.second;
        return std::sqrt(dx*dx + dy*dy);
    }
    scalar dist(const Index& I,const Index& I2) const {
        return dist(toGridIndex(I),toGridIndex(I2));
    }

    Index nx() const {
        return width;
    }
    Index ny() const {
        return height;
    }

    vec2 getGridSpace(grid_Index ij) const {
        scalar E = std::max(width,height);
        scalar u = (scalar(ij.first) - width*0.5)/E;
        scalar v = (scalar(ij.second) - height*0.5)/E;
        vec2 rslt = vec2(u,v)*2;
        return rslt;
    }

};

#endif // PERIODIC2DGRID_H
