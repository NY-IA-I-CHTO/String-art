import sys
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import imageio.v2 as imageio  # Используем imageio.v2 для избежания предупреждений
from skimage.transform import resize as imresize
from skimage.color import rgb2gray
import math
from collections import defaultdict
from bresenham import *


def image(filename, size):
    img = imresize(rgb2gray(imageio.imread(filename)), (size, size))
    return img


def build_arc_adjecency_matrix(n, radius):
    print("building sparse adjecency matrix")
    hooks = np.array([[math.cos(np.pi*2*i/n), math.sin(np.pi*2*i/n)] for i in range(n)])
    hooks = (radius * hooks).astype(int)
    edge_codes = []
    row_ind = []
    col_ind = []
    for i, ni in enumerate(hooks):
        for j, nj in enumerate(hooks[i+1:], start=i+1):
            edge_codes.append((i, j))
            pixels = bresenham(ni, nj).path
            edge = []
            for pixel in pixels:
                pixel_code = (pixel[1]+radius)*(radius*2+1) + (pixel[0]+radius)
                edge.append(pixel_code)
            row_ind += edge
            col_ind += [len(edge_codes)-1] * len(edge)
    sparse = scipy.sparse.csr_matrix(([1.0]*len(row_ind), (row_ind, col_ind)), 
                    shape=((2*radius+1)*(2*radius+1), len(edge_codes)))
    return sparse, hooks, edge_codes


def build_circle_adjecency_matrix(radius, small_radius):
    print("building sparse adjecency matrix")
    edge_codes = []
    row_ind = []
    col_ind = []
    pixels = circle(small_radius)
    for i, cx in enumerate(range(-radius+small_radius+1, radius-small_radius-1, 1)):
        for j, cy in enumerate(range(-radius+small_radius+1, radius-small_radius-1, 1)):
            edge_codes.append((i, j))
            edge = []
            for pixel in pixels:
                px, py = cx+pixel[0], cy+pixel[1]
                pixel_code = (py+radius)*(radius*2+1) + (px+radius)
                edge.append(pixel_code)
            row_ind += edge
            col_ind += [len(edge_codes)-1] * len(edge)
    sparse = scipy.sparse.csr_matrix(([1.0]*len(row_ind), (row_ind, col_ind)), 
                    shape=((2*radius+1)*(2*radius+1), len(edge_codes)))
    hooks = []
    return sparse, hooks, edge_codes


def build_image_vector(img, radius):
    assert img.shape[0] == img.shape[1]
    img_size = img.shape[0]
    row_ind = []
    col_ind = []
    data = []
    for y, line in enumerate(img):
        for x, pixel_value in enumerate(line):
            global_x = x - img_size//2
            global_y = y - img_size//2
            pixel_code = (global_y+radius)*(radius*2+1) + (global_x+radius)
            data.append(float(pixel_value))
            row_ind.append(pixel_code)
            col_ind.append(0)
    sparse_b = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), 
                    shape=((2*radius+1)*(2*radius+1), 1))
    return sparse_b


def reconstruct(x, sparse, radius):
    b_approx = sparse.dot(x)
    b_image = b_approx.reshape((2*radius+1, 2*radius+1))
    b_image = np.clip(b_image, 0, 255)
    return b_image


def reconstruct_and_save(x, sparse, radius, filename):
    brightness_correction = 1.2
    b_image = reconstruct(x * brightness_correction, sparse, radius)
    
    # Преобразуем изображение в 8-битное перед сохранением
    b_image = np.clip(b_image, 0, 255).astype(np.uint8)
    imageio.imwrite(filename, b_image)


def dump_arcs(solution, hooks, edge_codes, filename):
    with open(filename, "w") as f:
        n = len(hooks)
        print(n, file=f)
        for i, (x, y) in enumerate(hooks):
            print("%d\t%f\t%f" % (i, x, y), file=f)
        print(file=f)
        assert len(edge_codes) == len(solution)
        for (i, j), value in zip(edge_codes, solution):
            if value == 0:
                continue
            if value == int(value):
                value = int(value)
            print("%d\t%d\t%s" % (i, j, str(value)), file=f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py input_image output_prefix")
        sys.exit(1)
        
    filename, output_prefix = sys.argv[1:3]

    n = 180
    radius = 250

    sparse, hooks, edge_codes = build_arc_adjecency_matrix(n, radius)
    # sparse, hooks, edge_codes = build_circle_adjecency_matrix(radius, 10)

    shrinkage = 0.75
    img = image(filename, int(radius * 2 * shrinkage))
    sparse_b = build_image_vector(img, radius)

    print("solving linear system")
    result = scipy.sparse.linalg.lsqr(sparse, np.array(sparse_b.todense()).flatten())
    print("done")
    x = result[0]

    reconstruct_and_save(x, sparse, radius, output_prefix+"-allow-negative.png")

    x = np.clip(x, 0, 1e6)

    reconstruct_and_save(x, sparse, radius, output_prefix+"-unquantized.png")
    dump_arcs(x, hooks, edge_codes, output_prefix+"-unquantized.txt")

    quantization_level = 30
    clip_factor = 0.3
    if quantization_level is not None:
        max_edge_weight_orig = np.max(x)
        x_quantized = (x / np.max(x) * quantization_level).round()
        x_quantized = np.clip(x_quantized, 0, int(np.max(x_quantized) * clip_factor))
        x = x_quantized / quantization_level * max_edge_weight_orig
        dump_arcs(x_quantized, hooks, edge_codes, output_prefix+".txt")

    reconstruct_and_save(x, sparse, radius, output_prefix+".png")

    if quantization_level is not None:
        arc_count = 0
        total_distance = 0.0
        hist = defaultdict(int)
        for edge_code, multiplicity in enumerate(x_quantized):
            multiplicity = int(multiplicity)
            hist[multiplicity] += 1
            arc_count += multiplicity
            hook_index1, hook_index2 = edge_codes[edge_code]
            hook1, hook2 = hooks[hook_index1], hooks[hook_index2]
            distance = np.linalg.norm(hook1.astype(float) - hook2.astype(float)) / radius
            total_distance += distance * multiplicity
        for multiplicity in range(max(hist.keys())+1):
            print(multiplicity, hist[multiplicity])
        print("total arc count", arc_count)
        print("number of different arcs used", len(x_quantized[x_quantized>0]))
        print("total distance (assuming a unit diameter circle)", total_distance / 2)


if __name__ == "__main__":
    main()