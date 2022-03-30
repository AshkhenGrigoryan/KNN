# KNN
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
def get_neighbors_labels(X_train, y_train, x_new, k):
    distances = []
    for x in X_train:
        distances.append(euclidean_distance(x, x_new))
    nearest = np.argsort(distances)[:k]
    return y_train[nearest]
def get_response(neighbor_labels, num_classes=3):
  class_votes = np.zeros(num_classes)
    for label in neighbor_labels:
        class_votes[label] += 1
    return np.argmax(class_votes)
    
def compute_accuracy(y_pred, y_test):
     return np.mean(y_pred == y_test)
def predict(X_train, y_train, X_test, k):
 y_pred = []
    for x_new in X_test:
        neighbors = get_neighbors_labels(X_train, y_train, x_new, k)
        y_pred.append(get_response(neighbors))
    y_pred = np.array(y_pred)
    return y_pred
