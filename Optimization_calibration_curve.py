import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

filepath = r"C:\Users\kubad\Dropbox\Zlodějíčkovo slůj\Python\UVN_scripts\Film_dosimetry\Filmy\SB_2502_new\kalibrace.txt"


def load_channels_from_txt():
    """
    Reads data from a text file and splits the third column
    into three separate lists (R, G, B) by (index % 3).
    Returns flipped numpy arrays P_R, P_G, P_B.
    """
    global filepath
    R_vals, G_vals, B_vals = [], [], []

    with open(filepath, 'r') as f:
        counter = 0
        for line in f:
            cols = line.strip().split()
            val = float(cols[2])  # third column
            idx = counter

            if idx % 3 == 0:
                R_vals.append(val)
            elif idx % 3 == 1:
                G_vals.append(val)
            else:
                B_vals.append(val)

            counter += 1

    P_R = np.flip(np.array(R_vals))
    P_G = np.flip(np.array(G_vals))
    P_B = np.flip(np.array(B_vals))

    D_known = np.arange(len(P_B))
    return D_known, P_R, P_G, P_B

D_known, P_R, P_G, P_B = load_channels_from_txt()


def rational_fun(x, a, b, c):
    return a / (x + b) + c


def fit_individual_channels(dose, pixel_int):
    # Initial guess for parameters (all ones)
    x_data = pixel_int
    y_data = dose

    # Fit the rational function to the data
    params, params_covariance = curve_fit(rational_fun, x_data, y_data, p0=[115000, -8000, -2])

    # plt.scatter(x_data, y_data)
    # plt.plot(np.arange(min(x_data), max(x_data), 1000), rational_fun(np.arange(min(x_data), max(x_data), 1000), *params))
    # plt.show()

    return params


# Objective function with weights
def objective(w):
    D_known, P_R, P_G, P_B = load_channels_from_txt()

    weights = np.ones_like(D_known)
    w_R, w_G, w_B = w

    a_R, b_R, c_R = fit_individual_channels(D_known, P_R)
    a_G, b_G, c_G = fit_individual_channels(D_known, P_G)
    a_B, b_B, c_B = fit_individual_channels(D_known, P_B)

    D_calc = (w_R * rational_fun(P_R, a_R, b_R, c_R) + w_G * rational_fun(P_G, a_G, b_G, c_G)
              + w_B * rational_fun(P_B, a_B, b_B, c_B))

    error = weights * (D_known - D_calc) ** 2

    return np.sum(error)

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

#a_R, b_R, c_R = fit_individual_channels(D_known, P_R)
#print(f"Parametry červeného kanálu jsou: a = {a_R}, b = {b_R}, c = {c_R}")
#a_G, b_G, c_G = fit_individual_channels(D_known, P_G)
#print(f"Parametry zeleného kanálu jsou: a = {a_G}, b = {b_G}, c = {c_G}")
#a_B, b_B, c_B = fit_individual_channels(D_known, P_B)
#print(f"Parametry modrého kanálu jsou: a = {a_B}, b = {b_B}, c = {c_B}")


def main():
    # Constraints
    D_known, P_R, P_G, P_B = load_channels_from_txt()

    params_R = fit_individual_channels(D_known, P_R)
    params_G = fit_individual_channels(D_known, P_G)
    params_B = fit_individual_channels(D_known, P_B)

    # Initial guess for weights
    initial_weights = np.array([1/3, 1/3, 1/3])


    # Minimize the objective function with weights
    result = minimize(objective, initial_weights, constraints=constraints, bounds=[(0.1, 1), (0.05, 1), (0.05, 1)])

    # Optimal weights
    optimal_weights = result.x
    w_R, w_G, w_B = optimal_weights

    print(f"Optimal weights: w_R = {w_R}, w_G = {w_G}, w_B = {w_B}")
    print(f"{w_R:.4f} * ({params_R[0]:.4f} / (x + {params_R[1]:.4f}) + {params_R[2]:.4f}) + "
          f"{w_G:.4f} * ({params_G[0]:.4f} / (x + {params_G[1]:.4f}) + {params_G[2]:.4f}) + "
          f"{w_B:.4f} * ({params_B[0]:.4f} / (x + {params_B[1]:.4f}) + {params_B[2]:.4f})")


if __name__ == "__main__":
    main()





# Calibration data 0821
# D_known = np.flip(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))  # Known doses
# P_R = np.flip(np.array([48900, 36626.9, 30803.267, 27337.7895, 24820.363, 22634.145, 21365.856, 19635.655, 18610.419]))  # Red channel pixel values
# P_G = np.flip(np.array([48110, 41258.7485, 36509.769, 33096.754, 30651.9895, 28671.3335, 26818.625, 25422.3435, 24184.0855]))  # Green channel pixel values
# P_B = np.flip(np.array([36842, 35242.8385, 34260.286, 33038.879, 31816.6745, 31147.276, 30315.637, 29578.058, 28761.6035]))  # Blue channel pixel values

# # Calibration data 0903 (no color correction)
# D_known = np.flip(np.array([1, 2, 3, 4, 5, 6, 7, 8]))  # Known doses
# P_R = np.flip(np.array([32561.15, 26784.946, 23197.848, 20667.541, 18598.666, 17312.779, 15773.61, 14751.801]))
# P_G = np.flip(np.array([36729.582, 31549.405, 27946.034, 25316.718, 23275.692, 21511.152, 20041.608, 18833.88]))
# P_B = np.flip(np.array([32081.632, 29859.664, 28114.057, 26663.751, 25453.38, 24370.341, 23429.96, 22486.102]))

# # Calibration data 0310 (no color correction)
# D_known = np.flip(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))  # Known doses
#
# P_R = np.flip(np.array([45229.274, 33348.904, 27502.639, 23395.859, 20784.978, 18560.933, 17065.081, 15836.417, 14963.294]))
# P_G = np.flip(np.array([45777.193, 37689.16, 32609.383, 28859.109, 25918.751, 23865.127, 21995.531, 20549.775, 19344.211]))
# P_B = np.flip(np.array([35514.897, 32647.515, 30595.679, 28841.425, 27226.663, 26087.249, 24719.747, 23903.912, 22962.95]))

# Calibration data 1312
# D_known = np.flip(np.array([0, 1, 2, 3, 4, 5, 6]))  # Known doses

# Hodnoty pro jednotlivé kanály po trojicích
# P_R = np.flip(np.array([47668.857, 37657.459, 31692.294, 27269.807, 23913.151, 21658.360, 19723.748]))
# P_G = np.flip(np.array([46228.706, 40743.512, 36686.351, 33196.275, 30200.780, 28327.806, 26317.789]))
# P_B = np.flip(np.array([35176.138, 33033.940, 31572.830, 30150.318, 28835.981, 28111.118, 27193.964]))

# # # Calibration data 1212
# D_known = np.flip(np.array([0, 1, 2, 3, 4, 5, 6]))  # Known doses
#
# # Hodnoty pro jednotlivé kanály po trojicích
# P_R = np.flip(np.array([47686.857, 37443.040, 31674.413, 27346.160, 24173.785, 21666.481, 19552.510]))
# P_G = np.flip(np.array([46204.706, 40685.354, 36757.565, 33397.610, 30683.187, 28271.295, 26111.511]))
# P_B = np.flip(np.array([34802.138, 32769.690, 31464.369, 30154.402, 29142.737, 28020.649, 26953.815]))


# Calibration data 0320
# D_known = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0])
# P_R = np.array([16784.461, 17514.671, 18338.132, 19723.659, 21160.815, 22559.505, 24632.584, 27424.570, 31162.886, 37506.510, 42187.068, 49923.147])
# P_G = np.array([22203.654, 23353.642, 24530.136, 25796.388, 27169.762, 29069.388, 31079.905, 33849.823, 37187.135, 41904.590, 45177.596, 48856.908])
# P_B = np.array([27799.684, 28385.368, 29020.640, 29985.294, 30677.544, 31349.777, 32447.213, 33563.986, 34808.877, 35946.925, 36712.106, 37120.027])
