import torch
import torch.optim as optim
import torch.nn as nn


# Define the objective function
def objective(p1, p2, path, f, eps):
    # Compute the distances between consecutive x's
    dist_1 = torch.unsqueeze(torch.norm(p1 - path[1, :].reshape(p1.shape)), dim=-1)
    dists = torch.norm(path[1:, :] - path[:-1, :], dim=1)
    dist_n = torch.unsqueeze(torch.norm(p2 - path[-1, :].reshape(p2.shape)), dim=-1)
    dists = torch.cat((dist_1, dists, dist_n), dim=0)
    # values of the integrand
    value_1 = f(torch.norm((path[0] / 2 + p1.T / 2).T, dim=0, keepdim=True))
    f_values = f(torch.norm(path.T[:, :-1] / 2 + path.T[:, 1:] / 2, dim=0, keepdim=True))
    value_n = f(torch.norm((path[-1] / 2 + p2.T / 2).T, dim=0, keepdim=True))
    f_values = torch.cat((value_1, f_values, value_n), dim=1)
    # Compute the line integral
    line_integral = (dists * f_values).sum()
    # Compute the penalty for violating the distance constraint
    dist_violations = torch.relu(dists - eps)
    penalty = torch.sum(dist_violations)

    # Return the objective function and penalty as a tuple
    return line_integral, penalty


def path_between_two_points(a, b, n):
    d = a.shape[0]
    # Initialize the path points to be regularly placed on the cord between a and b
    x = torch.linspace(0, 1, n, dtype=torch.float).view(n, 1).repeat(1, d).to("cuda")
    x = x * (b.t() - a.t()) + a.t()
    x = x + torch.rand(x.shape).to("cuda") / 100
    x = nn.Parameter(x[1:-1, :])  # without a and b(stay in place)
    return x


def norm_aware_interpolation(f, a, b, n, eps, eta=0.001):
    x = path_between_two_points(a, b, n)

    optimizer = optim.Adam([x], lr=0.01)
    lam = 1
    last_loss = torch.inf
    diff_counter = 0
    for i in range(10000):
        optimizer.zero_grad()
        line_integral, penalty = objective(a, b, x, f, eps)
        loss = -1 * line_integral + lam * penalty
        loss.backward()
        optimizer.step()

        # early stopping
        loss_diff = torch.abs(last_loss - loss)
        if loss_diff < eta:
            diff_counter += 1
        else:
            diff_counter = 0
        last_loss = loss

        if diff_counter == 10:
            break

        if i % 10 == 0:
            print("Iteration {}, Loss: {:.4f}, Penalty: {:.4f}".format(
                i, loss.item(), penalty.item()))

    return x.detach()


def norm_aware_centroid_optimization(pdf_func, init_points, n_points_per_path, eps, init_c=None, eta=0.01):
    if init_c is not None:
        centroid = init_c
    else:
        centroid = torch.stack(init_points).mean(dim=0)
    if (centroid == 0).all():
        centroid = torch.rand(centroid.shape).to("cuda") / 100
    centroid = nn.Parameter(centroid)

    # create a path between the centroid and each initial point
    paths_ls = [path_between_two_points(point, centroid, n_points_per_path) for point in init_points]

    optimizer = optim.Adam(paths_ls + [centroid], lr=0.01)

    lamda = 1
    last_loss = torch.inf
    diff_counter = 0
    for i in range(10000):
        optimizer.zero_grad()
        # line integral for each path
        total_line_integral, total_penalty = [], []
        for idx, point in enumerate(init_points):
            # call objective function
            line_integral, penalty = objective(point, centroid, paths_ls[idx], pdf_func, eps)
            total_line_integral += [line_integral]
            total_penalty += [penalty]
        total_line_integral, total_penalty = torch.stack(total_line_integral), torch.stack(total_penalty)
        total_line_integral = total_line_integral.sum()
        total_penalty = total_penalty.sum()

        # negative log likelihood loss + penalty
        loss = -1 * total_line_integral - pdf_func(torch.norm(centroid)) + lamda * total_penalty
        loss.backward()
        optimizer.step()

        # check early stopping
        loss_diff = torch.abs(last_loss - loss)
        if loss_diff < eta:
            diff_counter += 1
        else:
            diff_counter = 0
        last_loss = loss

        if diff_counter == 5:
            break

        if i % 10 == 0:
            print("Iteration {}, Loss: {:.4f}, Penalty: {:.4f}".format(
                i, loss.item(), total_penalty.item()))

    paths_ls = list(map(lambda x: x.detach(), paths_ls))
    return centroid.detach(), paths_ls
