#include <iostream>

#include "isolation_forest.hxx"
#include "data.hxx"

int main()
{
    IsolationForest iforest(20, 64);
    iforest.fit(example_data);
    std::cout << iforest.score({ 17.0, 87.0, 12.0 }) << std::endl;
    for (auto row : example_data)
    {
        std::cout << iforest.score(row) << '\n';
    }
    return 0;
}