import numpy as np

NUM_SAMPLES = 100


def sample_theta():
    return np.random.uniform(np.pi / 5 - np.pi / 36, np.pi / 5 + np.pi / 36, NUM_SAMPLES)


def sample_d():
    return np.random.normal(5, 1, NUM_SAMPLES)


def transit(x, y):
    theta, d = sample_theta(), sample_d()

    xp = x + d * np.cos(theta)
    yp = y + d * np.sin(theta)

    return xp, yp


def compute_da(x, y):
    return np.sqrt(np.square(-100 - x) + np.square(100 - y))


def compute_db(x, y):
    return np.sqrt(np.square(150 - x) + np.square(90 - y))


def normal_pdf(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.square((x - mean) / std))


def normalize(x):
    return x / np.sum(x)


def run_particle_filter():
    results = 'x mean & y mean & x variance & y variance\n'
    for i in range(4):
        results += f'Run#{i+1}\n\n'
        x = np.random.normal(0, 1, NUM_SAMPLES)
        y = np.random.normal(0, 1, NUM_SAMPLES)

        observations = [[143.69345025, 166.824055471],
                        [145.664295064, 164.501752829],
                        [144.591594567, 157.474359865],
                        [146.50065381, 152.469730508],
                        [148.997244853, 145.093420417],
                        [148.636960211, 142.497183979],
                        [152.357648068, 135.919999894],
                        [152.367509975, 133.667699492],
                        [155.708667162, 126.696182098],
                        [159.03213926, 122.950895656],
                        [159.068406295, 116.296800819],
                        [164.570101706, 111.140841502],
                        [164.408405506, 107.94706493],
                        [170.250724817, 101.808614448],
                        [171.489734721, 96.7475872389],
                        [173.631500934, 91.6295157657],
                        [178.451332801, 85.3706909502],
                        [182.747629193, 80.8448050049],
                        [183.013423256, 78.7848074168],
                        [185.554608457, 74.7064611403]]

        for (ra, rb) in observations:
            x, y = transit(x, y)
            da, db = compute_da(x, y), compute_db(x, y)
            weights = normal_pdf(ra, da, 1) * normal_pdf(rb, db, 1)
            weights = normalize(weights)

            indices = np.random.choice(NUM_SAMPLES, size=NUM_SAMPLES, p=weights)
            x, y = x[indices], y[indices]
            results += f'{np.mean(x)} & {np.mean(y)} & {np.var(x)} & {np.var(y)}\n'
        results += '\n'

    with open('particle_filter_messy_results.txt', 'w+') as fh:
        fh.write(results)


run_particle_filter()
