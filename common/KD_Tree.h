#ifndef OCTREE_H
#define OCTREE_H

#include "types.h"
#include <bitset>

class KdTree {
public:

    struct Element : public Vec {
        Element(const Vec& x,size_t i) : Vec(x),id(i) {}
        Element() {}
        size_t id;
    };

    size_t dim;
    int n = 0;
    Vec center;
    Element value;

    size_t id;
    std::vector<KdTree*> nodes;
    bool is_leaf;
    double scale;
    double factor;

    ~KdTree() {
        for (auto child : nodes)
            if (child)
                delete child;
    }

    size_t nchild() const {
        return 1 << dim;
    }


    /**
     * @brief KdTree accelerating structure using octree system, garanting
     * log(n) insertion and query
     * @param f scale factor (will work optimally with objects contained
     * between [-f,f]^3
     */
    KdTree(size_t dim,double f = 1) : dim(dim),n(0),is_leaf(true),factor(f) {
        center = Vec::Zero(dim);
        scale = 1.;

        nodes.resize(nchild(),nullptr);
    }

private:
    KdTree(size_t dim,const Vec &C,double f, size_t h) : dim(dim), n(h),factor(f) {
        scale = 1./double( 1 << n);
        nodes.resize(nchild(),nullptr);
        is_leaf = true;
        center = C;
    }

    KdTree* leaf(const Vec& C) {
        KdTree* L = new KdTree(dim,C,factor,n+1);
        L->is_leaf = true;
        return L;
    }

    double getScale() const {return scale;}

    double getChildrenScale() const {return scale*0.5;}

    struct bitchain : std::vector<bool> {
        bitchain(int s) {
            resize(s,false);
        }

        bitchain(int size,int value) {
            //init with value, with bit decomposition
            resize(size);
            for (int i = 0;i<size;i++)
                operator[](i) = value & (1 << i);
        }

        size_t to_ulong() const {
            size_t res = 0;
            for (size_t i = 0;i<size();i++)
                res += (1 << i) * operator[](i);
            return res;
        }
    };

    void initChildren() {
        is_leaf = false;
        for (int c = 0;c<nchild();c++){
            bitchain bs(dim,c);
            Vec offset = Vec::Zero(dim);
            for (int b = 0;b<dim;b++)
                offset[b] = bs[b] ? 1 : -1;
            nodes[c] = leaf(center + offset * getChildrenScale());
        }
    }

    inline double distanceToValue(const Vec& x) const {
        return (value - x).norm()/factor;
    }

    Element nearest(const Vec& x,Element& best,double& d) const {
        if (is_leaf)
            return best;
        for (int i = 0;i<dim;i++)
            if (x[i]/factor < center[i] - getScale() - d || x[i]/factor > center[i] + getScale() + d)
                return best;
        double cd = distanceToValue(x);
        if (cd < d){
            best = value;
            d = cd;
        }
        bitchain bits(dim);
        for (size_t i = 0;i<dim;i++)
            bits[i] = x[i] > factor*center[i];
        int code = bits.to_ulong();
        for (int i = 0;i<nchild();i++)
            best = nodes[i ^ code]->nearest(x,best,d);
        return best;
    }

    // AcceleratingStructure interface
public:
    void addElement(const Element& x) {
        if (is_leaf){
            value = x;
            initChildren();
            return;
        }
        bitchain bits(dim);
        for (size_t i = 0;i<dim;i++)
            bits[i] = x[i] > factor*center[i];
        nodes[bits.to_ulong()]->addElement(x);
    }

    Element nearestNeighboor(const Vec &x) const {
        double distance = 1e6;
        Element E;
        auto NN = nearest(x,E,distance);
        return E;
    }

    int nbElements() const{
        if (is_leaf)
            return 0;
        int s = 1;
        for (auto c : nodes)
            s += c->nbElements();
        return s;
    }
};
#endif // OCTREE_H
