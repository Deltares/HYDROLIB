# The scripts are credited to Dr. Cho
# The source of the scripts: https://here.isnew.info/strahler-stream-order-in-python.html

from osgeo import ogr


def find_head_lines(lines):
    head_idx = []

    num_lines = len(lines)
    for i in range(num_lines):
        line = lines[i]
        first_point = line[0]

        has_upstream = False

        for j in range(num_lines):
            if j == i:
                continue
            line = lines[j]
            last_point = line[len(line) - 1]

            if first_point == last_point:
                has_upstream = True

        if not has_upstream:
            head_idx.append(i)

    return head_idx


def find_prev_lines(curr_idx, lines):
    prev_idx = []

    num_lines = len(lines)

    line = lines[curr_idx]
    first_point = line[0]

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        last_point = line[len(line) - 1]

        if first_point == last_point:
            prev_idx.append(i)

    return prev_idx


def find_next_line(curr_idx, lines):
    num_lines = len(lines)

    line = lines[curr_idx]
    last_point = line[len(line) - 1]

    next_idx = None

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        first_point = line[0]

        if last_point == first_point:
            next_idx = i
            break

    return next_idx


def find_sibling_line(curr_idx, lines):
    num_lines = len(lines)

    line = lines[curr_idx]
    last_point = line[len(line) - 1]

    sibling_idx = None

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        last_point2 = line[len(line) - 1]

        if last_point == last_point2:
            sibling_idx = i
            break

    return sibling_idx
