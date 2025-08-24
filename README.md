myModelSimulation.py simulates the battery behaviour

nmc_20Ah.py and nmc_SANYOUR18650.py using the same format of NREL model parameters and formulas developed based on the paper: https://www.sciencedirect.com/science/article/pii/S2352152X23014135?via%3Dihub

nmc111_samsung94ah.py uses a semi-empirical model to fit the aging test found in the paper: https://www.researchgate.net/publication/347019356_Analysis_of_Lithium-ion_Battery_Cells_Degradation_Based_on_Different_Manufacturers nmc111_samsung94ah_SR.py uses a symbolic regression model. Both semi-empirical and symbolic regression models are using the method in the paper: https://www.sciencedirect.com/science/article/pii/S2352152X23024404?via%3Dihub
