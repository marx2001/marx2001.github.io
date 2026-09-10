
# Python 3 version
with open("k-points", 'r+') as file:
    lines = file.readlines()

line_count = len(lines)
mesh = 20
print("k-points along high symmetry lines")
print((line_count - 1) * mesh + 1)
print("Reciprocal")

# Convert each line to a list of floats
for i in range(line_count):
    lines[i] = lines[i].strip().split()
    lines[i][0] = float(lines[i][0])
    lines[i][1] = float(lines[i][1])
    lines[i][2] = float(lines[i][2])

# Generate the k-point mesh
for i in range(line_count - 1):
    for j in range(mesh):
        # Calculate the k-point coordinates for the current position along the path
        for k in range(3):
            print(lines[i][k] + (lines[i + 1][k] - lines[i][k]) * j / (mesh + 0.0), end=' ')
        print(1.0)

# Print the last k-point
print(lines[line_count - 1][0], lines[line_count - 1][1], lines[line_count - 1][2], "1.0")
