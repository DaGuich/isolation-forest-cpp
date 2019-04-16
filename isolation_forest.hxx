//
// Created by matthias on 13.04.19.
//

#ifndef BOSWATCH_ISOLATION_FOREST_HXX
#define BOSWATCH_ISOLATION_FOREST_HXX

#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <type_traits>
#include <cassert>

template<class T>
using MatrixRow = std::vector<T>;
template<class T>
using Matrix = std::vector<MatrixRow<T>>;

template<class T = float>
class IsolationForest
{
    class IsolationTree
    {
        size_t m_current_height;
        size_t m_height_limit;
        size_t m_size;
        T m_split_value;
        unsigned int m_split_feature;
        std::shared_ptr<std::mt19937> pRand_gen;

        std::unique_ptr<IsolationTree> pLeft = nullptr;
        std::unique_ptr<IsolationTree> pRight = nullptr;

    public:
        IsolationTree(const std::shared_ptr<std::mt19937> &rand_gen,
                      size_t current_height,
                      size_t height_limit);

        void fit(const Matrix<T> &data);

        double path_length(const MatrixRow<T> &data);

        constexpr static double fact_c(size_t size);

    };

    size_t m_n_trees;
    size_t m_subsampling_size;
    size_t m_height_limit;
    std::mt19937 m_rand_gen;
    std::vector<std::shared_ptr<IsolationTree>> m_trees;

public:
    IsolationForest(size_t n_trees, size_t subsampling_size, size_t height_limit = 40);

    void fit(const Matrix<T> &data);

    double score(const MatrixRow<T> &data);
};

template<class T>
IsolationForest<T>::IsolationForest(size_t n_trees, size_t subsampling_size, size_t height_limit)
{
    m_n_trees = n_trees;
    m_subsampling_size = subsampling_size;
    m_height_limit = height_limit;
    std::random_device rd;
    m_rand_gen = std::mt19937(rd());
}

template<class T>
void IsolationForest<T>::fit(const Matrix<T> &data)
{
    assert(m_n_trees <= (data.size() / m_subsampling_size));

    auto pRandGen = std::make_shared<std::mt19937>(m_rand_gen);

    Matrix<T> shuffled_data(data);
    std::random_shuffle(shuffled_data.begin(), shuffled_data.end());

    for (auto i = 0; i < m_n_trees; ++i) {
        Matrix<T> tree_data;
        auto start_it = std::next(
                shuffled_data.cbegin(),
                m_subsampling_size * i);
        auto end_it = std::next(
                shuffled_data.cbegin(),
                m_subsampling_size * i + m_subsampling_size);
        tree_data.resize(m_subsampling_size);
        std::copy(start_it, end_it, tree_data.begin());
        std::shared_ptr<IsolationTree> tree = std::make_shared<IsolationTree>(
                IsolationTree(pRandGen, 0, m_height_limit));
        tree->fit(tree_data);
        m_trees.push_back(tree);
    }
}

template<class T>
double IsolationForest<T>::score(const MatrixRow<T> &data)
{
    auto sum = [](const std::vector<double> &v) constexpr -> double{
            double res = 0;
            for (const auto & val : v)
            {
                res += val;
            }
            return res;
    };

    std::vector<double> heights;
    for (auto tree : m_trees) {
        auto pl = tree->path_length(data);
        heights.push_back(pl);
    }
    auto summed_heights = sum(heights);
    return std::pow(2.0, -1 * (summed_heights / (IsolationTree::fact_c(m_subsampling_size) * m_n_trees)));
}

template<class T>
constexpr double IsolationForest<T>::IsolationTree::fact_c(const size_t size)
{
    auto n = static_cast<double>(size);
    auto H = [](double n) constexpr -> double{
            constexpr double euler_constant = 0.5772156649;
            return std::log(static_cast<double>(n)) + euler_constant;
    };
    return 2.0 * H(n - 1) - ((2.0 * (n - 1) / n));
}

template<class T>
double IsolationForest<T>::IsolationTree::path_length(const MatrixRow<T> &data)
{
    if (pLeft == nullptr || pRight == nullptr) {
        if (m_size < 2)
        {
            return static_cast<double>(m_current_height);
        }
        return IsolationTree::fact_c(m_size) + m_current_height;
    } else if (data[m_split_feature] < m_split_value) {
        return pLeft->path_length(data);
    } else {
        return pRight->path_length(data);
    }
}

template<class T>
IsolationForest<T>::IsolationTree::IsolationTree(const std::shared_ptr<std::mt19937> &rand_gen,
                                                 size_t current_height,
                                                 size_t height_limit)
{
    m_current_height = current_height;
    m_height_limit = height_limit;
    m_size = 0;
    m_split_feature = 0;
    pRand_gen = rand_gen;
    pLeft = nullptr;
    pRight = nullptr;
}

template<class T>
void IsolationForest<T>::IsolationTree::fit(const Matrix<T> &data)
{
    size_t n_features;
    T valmin, valmax;
    Matrix<T> left_data;
    Matrix<T> right_data;

    auto filter = [data, &left_data, &right_data](unsigned int feature, T value) {
        for (const auto &row : data) {
            if (row[feature] < value) {
                left_data.push_back(row);
            } else {
                right_data.push_back(row);
            }
        }
    };

    auto get_minmax = [&valmin, &valmax, data](unsigned int feature) {
        bool first = true;
        T current;
        for (const auto &row : data) {
            current = row[feature];
            if (first) {
                valmin = current;
                valmax = current;
                first = false;
            } else {
                if (valmin < current) {
                    valmin = current;
                }

                if (valmax > current) {
                    valmax = current;
                }
            }
        }
    };

    constexpr auto get_n_features = [](const Matrix<T> &data) constexpr -> size_t{
            size_t n_features = 0;
            bool first = true;
            for (const auto &row : data)
            {
                if (first) {
                    first = false;
                    n_features = row.size();
                }

                assert(n_features == row.size());
            }
            return n_features;
    };

    m_size = data.size();

    if ((m_size <= 1) || (m_height_limit <= m_current_height)) {
        pLeft = nullptr;
        pRight = nullptr;
        return;
    }

    n_features = get_n_features(data);

    {
        std::uniform_int_distribution<unsigned int> int_dist(0, n_features);
        m_split_feature = int_dist(*pRand_gen);
        get_minmax(m_split_feature);
    }

    if constexpr(std::is_integral<T>::value) {
        std::uniform_int_distribution<T> val_dist(valmin, valmax);
        m_split_value = val_dist(*pRand_gen);
    } else if constexpr(std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> val_dist(valmin, valmax);
        m_split_value = val_dist(*pRand_gen);
    }

    filter(m_split_feature, m_split_value);

    pLeft = std::make_unique<IsolationTree>(IsolationTree(pRand_gen, m_current_height + 1, m_height_limit));
    pRight = std::make_unique<IsolationTree>(IsolationTree(pRand_gen, m_current_height + 1, m_height_limit));
    pLeft->fit(left_data);
    pRight->fit(right_data);
}


#endif //BOSWATCH_ISOLATION_FOREST_HXX
