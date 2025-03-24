import numpy as np
import matplotlib.pyplot as plt

# Simulating two classes of data
np.random.seed(42)
class1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
class2 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100)

plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')
plt.legend()
plt.title('Simulated Data for LDA')
plt.show()

# Combine classes and create labels
X = np.vstack((class1, class2))
y = np.hstack((np.zeros(100), np.ones(100)))

# Separate data by class
class1 = X[y == 0]
class2 = X[y == 1]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Number of samples in Class 1:", class1.shape[0])
print("Number of samples in Class 2:", class2.shape[0])

# Calculate class means
mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2, axis=0)

print("Mean of Class 1:", mean1)
print("Mean of Class 2:", mean2)

# Visualize class means
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')
plt.scatter(mean1[0], mean1[1], color='red', s=200, marker='*', label='Mean Class 1')
plt.scatter(mean2[0], mean2[1], color='green', s=200, marker='*', label='Mean Class 2')
plt.legend()
plt.title('Class Means')
plt.show()

# Compute within-class scatter matrix
S_W = np.zeros((2, 2))
for c, mean in [(class1, mean1), (class2, mean2)]:
    S_c = np.zeros((2, 2))
    for sample in c:
        diff = (sample - mean).reshape(2, 1)
        S_c += np.dot(diff, diff.T)
    S_W += S_c

print("Within-class scatter matrix:")
print(S_W)

# Compute overall mean
mean_overall = np.mean(X, axis=0)

# Compute between-class scatter matrix
n1, n2 = class1.shape[0], class2.shape[0]
diff1 = (mean1 - mean_overall).reshape(2, 1)
diff2 = (mean2 - mean_overall).reshape(2, 1)
S_B = n1 * np.dot(diff1, diff1.T) + n2 * np.dot(diff2, diff2.T)

print("Between-class scatter matrix:")
print(S_B)

# Solve the generalized eigenvalue problem
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Sort eigenvectors by eigenvalues in descending order
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Select the eigenvector with the largest eigenvalue
w = eigvecs[:, 0]

print("Largest eigenvalue:", eigvals[0])
print("Corresponding eigenvector:", w)

# Project data onto LDA direction
X_projected = X.dot(w)

# Visualize projected data
plt.scatter(X_projected[y == 0], np.zeros(n1), label='Class 1')
plt.scatter(X_projected[y == 1], np.zeros(n2), label='Class 2')
plt.legend()
plt.title('Data Projected onto LDA Direction')
plt.xlabel('LDA Component')
plt.yticks([])
plt.show()

def lda_classify(x, w, mean1, mean2):
    # Project class means onto LDA direction
    m1 = w.dot(mean1)
    m2 = w.dot(mean2)
    
    # Compute midpoint
    midpoint = (m1 + m2) / 2
    
    # Project data point
    projection = w.dot(x)
    
    # Classify based on which side of the midpoint the projection falls
    return 0 if projection < midpoint else 1

# Test the classifier
test_point = np.array([1, 1])
prediction = lda_classify(test_point, w, mean1, mean2)
print(f"Test point {test_point} classified as Class {prediction}")

# Visualize decision boundary
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')
plt.scatter(test_point[0], test_point[1], color='red', s=200, marker='x', label='Test Point')

# Plot decision boundary
boundary_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
boundary_y = -(w[0] * boundary_x + (w.dot(mean1 + mean2) / 2)) / w[1]
plt.plot(boundary_x, boundary_y, 'k--', label='Decision Boundary')

plt.legend()
plt.title('LDA Classification')
plt.show()